"""Repetition diagnostics for generated concept-curriculum text."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


TOKEN_RE = re.compile(r"\b[\w'-]+\b")


def tokenize_text(text: str, *, lowercase: bool = True) -> list[str]:
    """Tokenize text for lightweight repetition diagnostics."""

    tokens = TOKEN_RE.findall(str(text))
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(tokens) < n:
        return []
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _document_id(row: dict[str, Any], index: int) -> str:
    value = row.get("id")
    if value is None:
        value = row.get("document_id")
    return str(value) if value is not None else f"row_{index:04d}"


def _text_for_row(row: dict[str, Any], *, text_field: str) -> str:
    value: Any = row
    for part in text_field.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return str(value or "")


def _field_value(row: dict[str, Any], field: str) -> str:
    if "." in field:
        value: Any = row
        for part in field.split("."):
            if not isinstance(value, dict):
                return ""
            value = value.get(part)
        return str(value) if value is not None else ""

    if field in row:
        return str(row.get(field))
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and field in metadata:
        return str(metadata.get(field))
    return ""


def _empty_ngram_report(*, min_n: int, max_n: int) -> dict[str, Any]:
    return {
        "summary": {
            str(n): {
                "total_ngrams": 0,
                "unique_ngrams": 0,
                "repeated_unique_ngrams": 0,
                "repeated_occurrences": 0,
                "repetition_ratio": 0.0,
            }
            for n in range(min_n, max_n + 1)
        },
        "top_ngrams": {str(n): [] for n in range(min_n, max_n + 1)},
    }


def _ngram_report_for_rows(
    rows: list[dict[str, Any]],
    *,
    min_n: int,
    max_n: int,
    min_count: int,
    min_documents: int,
    top_k: int,
    sample_documents: int,
    text_field: str,
    lowercase: bool,
) -> dict[str, Any]:
    if min_n <= 0:
        raise ValueError("min_n must be positive.")
    if max_n < min_n:
        raise ValueError("max_n must be greater than or equal to min_n.")
    if min_count <= 0:
        raise ValueError("min_count must be positive.")
    if min_documents <= 0:
        raise ValueError("min_documents must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if sample_documents <= 0:
        raise ValueError("sample_documents must be positive.")

    if not rows:
        return _empty_ngram_report(min_n=min_n, max_n=max_n)

    counts_by_n: dict[int, Counter[str]] = {n: Counter() for n in range(min_n, max_n + 1)}
    docs_by_n: dict[int, dict[str, set[str]]] = {
        n: defaultdict(set) for n in range(min_n, max_n + 1)
    }
    total_by_n: dict[int, int] = {n: 0 for n in range(min_n, max_n + 1)}

    for index, row in enumerate(rows):
        doc_id = _document_id(row, index)
        tokens = tokenize_text(_text_for_row(row, text_field=text_field), lowercase=lowercase)
        for n in range(min_n, max_n + 1):
            doc_ngrams = _ngrams(tokens, n)
            if not doc_ngrams:
                continue
            counts_by_n[n].update(doc_ngrams)
            total_by_n[n] += len(doc_ngrams)
            for ngram in set(doc_ngrams):
                docs_by_n[n][ngram].add(doc_id)

    summary: dict[str, dict[str, int | float]] = {}
    top_ngrams: dict[str, list[dict[str, Any]]] = {}

    for n in range(min_n, max_n + 1):
        counts = counts_by_n[n]
        total = total_by_n[n]
        repeated_unique = sum(
            1
            for ngram, count in counts.items()
            if count >= min_count and len(docs_by_n[n][ngram]) >= min_documents
        )
        repeated_occurrences = sum(
            count - 1
            for ngram, count in counts.items()
            if count >= min_count and len(docs_by_n[n][ngram]) >= min_documents
        )

        summary[str(n)] = {
            "total_ngrams": total,
            "unique_ngrams": len(counts),
            "repeated_unique_ngrams": repeated_unique,
            "repeated_occurrences": repeated_occurrences,
            "repetition_ratio": round(repeated_occurrences / total, 6) if total else 0.0,
        }

        candidates = [
            (ngram, count, len(docs_by_n[n][ngram]), sorted(docs_by_n[n][ngram]))
            for ngram, count in counts.items()
            if count >= min_count and len(docs_by_n[n][ngram]) >= min_documents
        ]
        candidates.sort(key=lambda item: (-item[2], -item[1], item[0]))
        top_ngrams[str(n)] = [
            {
                "ngram": ngram,
                "count": count,
                "document_count": document_count,
                "document_ids": document_ids[:sample_documents],
            }
            for ngram, count, document_count, document_ids in candidates[:top_k]
        ]

    return {"summary": summary, "top_ngrams": top_ngrams}


def build_repetition_report(
    rows: list[dict[str, Any]],
    *,
    min_n: int = 3,
    max_n: int = 6,
    min_count: int = 2,
    min_documents: int = 2,
    top_k: int = 25,
    sample_documents: int = 5,
    text_field: str = "text",
    lowercase: bool = True,
    group_by: list[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable repeated n-gram report."""

    token_counts = [
        len(tokenize_text(_text_for_row(row, text_field=text_field), lowercase=lowercase))
        for row in rows
    ]
    report = {
        "document_count": len(rows),
        "token_count": sum(token_counts),
        "settings": {
            "min_n": min_n,
            "max_n": max_n,
            "min_count": min_count,
            "min_documents": min_documents,
            "top_k": top_k,
            "sample_documents": sample_documents,
            "text_field": text_field,
            "lowercase": lowercase,
            "group_by": list(group_by or []),
        },
        **_ngram_report_for_rows(
            rows,
            min_n=min_n,
            max_n=max_n,
            min_count=min_count,
            min_documents=min_documents,
            top_k=top_k,
            sample_documents=sample_documents,
            text_field=text_field,
            lowercase=lowercase,
        ),
    }

    if group_by:
        grouped_report: dict[str, dict[str, Any]] = {}
        for field in group_by:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[_field_value(row, field) or "<missing>"].append(row)
            grouped_report[field] = {
                group_value: {
                    "document_count": len(group_rows),
                    **_ngram_report_for_rows(
                        group_rows,
                        min_n=min_n,
                        max_n=max_n,
                        min_count=min_count,
                        min_documents=min_documents,
                        top_k=top_k,
                        sample_documents=sample_documents,
                        text_field=text_field,
                        lowercase=lowercase,
                    ),
                }
                for group_value, group_rows in sorted(groups.items())
            }
        report["groups"] = grouped_report

    return report
