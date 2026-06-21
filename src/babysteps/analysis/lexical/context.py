"""Context-neighbor helpers for lexical analysis reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from babysteps.analysis.lexical.targets import normalize_lexeme


# Keep negators such as "not", "no", and "never" out of the default list:
# they are often useful context-quality signals around evaluation concepts.
DEFAULT_CONTEXT_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "around",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}.")


def _normalize_stopwords(values: Iterable[Any]) -> set[str]:
    return {
        normalized
        for value in values
        if (normalized := normalize_lexeme(value))
    }


def _word_values(raw_value: Any) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        return _normalize_stopwords(raw_value.replace(",", " ").split())
    if isinstance(raw_value, Iterable):
        return _normalize_stopwords(raw_value)
    raise TypeError("Expected a stopword string or list of words.")


def _stopword_values(raw_value: Any) -> set[str]:
    if raw_value is None:
        return set(DEFAULT_CONTEXT_STOPWORDS)
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"", "default", "english"}:
            return set(DEFAULT_CONTEXT_STOPWORDS)
        if value in {"none", "off", "false"}:
            return set()
        return _word_values(value)
    if isinstance(raw_value, Iterable):
        return _normalize_stopwords(raw_value)
    raise TypeError(
        "analysis.context_neighbor_stopwords must be 'english', 'none', "
        "or a list of words."
    )


def context_neighbor_stopwords_from_config(
    analysis_config: dict[str, Any],
) -> frozenset[str]:
    """Build the stopword set used for content-neighbor context summaries."""

    enabled = _as_bool(
        analysis_config.get("filter_context_neighbor_stopwords"),
        default=True,
    )
    if not enabled:
        return frozenset()

    stopwords = _stopword_values(analysis_config.get("context_neighbor_stopwords"))
    stopwords.update(
        _word_values(analysis_config.get("extra_context_neighbor_stopwords"))
    )
    stopwords.difference_update(
        _word_values(analysis_config.get("keep_context_neighbor_stopwords"))
    )
    return frozenset(stopwords)


def filter_counter_keys(
    counter: Counter[str],
    excluded_values: set[str] | frozenset[str],
) -> Counter[str]:
    """Return a copy of a counter without excluded normalized words."""

    if not excluded_values:
        return Counter(counter)
    return Counter(
        {
            value: count
            for value, count in counter.items()
            if value not in excluded_values
        }
    )


__all__ = [
    "DEFAULT_CONTEXT_STOPWORDS",
    "context_neighbor_stopwords_from_config",
    "filter_counter_keys",
]
