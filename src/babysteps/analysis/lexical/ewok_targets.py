"""Export EWoK concept columns into lexical-analysis target files."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from babysteps.analysis.lexical.targets import normalize_lexeme


EWOK_CONCEPT_FIELDS = ("ConceptA", "ConceptB")
PLACEHOLDER_VALUES = {"", "-"}
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class EwokConceptSummary:
    """Aggregated EWoK concept usage within one domain."""

    domain: str
    concept: str
    eval_count: int = 0
    field_counts: Counter[str] = field(default_factory=Counter)
    source_files: set[str] = field(default_factory=set)

    @property
    def reason_unsupported(self) -> str | None:
        if is_placeholder_concept(self.concept):
            return "placeholder"
        if is_multiword_concept(self.concept):
            return "multiword"
        return None

    @property
    def is_supported(self) -> bool:
        return self.reason_unsupported is None


def concept_key(value: Any) -> str:
    """Normalize one EWoK concept label for lexical target export."""

    if value is None:
        return ""
    text = str(value).replace("_", " ")
    return normalize_lexeme(text)


def is_placeholder_concept(value: str) -> bool:
    return concept_key(value) in PLACEHOLDER_VALUES


def is_multiword_concept(value: str) -> bool:
    return bool(re.search(r"\s+", concept_key(value)))


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", concept_key(value)).strip("_")
    return slug or "concept"


def read_ewok_concepts(ewok_dir: str | Path) -> list[EwokConceptSummary]:
    """Read all EWoK JSONL files and aggregate ConceptA/ConceptB by domain."""

    ewok_dir = Path(ewok_dir)
    if not ewok_dir.exists():
        raise FileNotFoundError(f"Missing EWoK directory: {ewok_dir}")

    paths = sorted(ewok_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No EWoK JSONL files found in: {ewok_dir}")

    grouped: dict[tuple[str, str], EwokConceptSummary] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object at {path}:{line_number}.")
                domain = concept_key(row.get("Domain", path.stem)) or path.stem
                for field_name in EWOK_CONCEPT_FIELDS:
                    raw_concept = row.get(field_name, "")
                    key = concept_key(raw_concept)
                    if not key and str(raw_concept).strip() not in PLACEHOLDER_VALUES:
                        continue
                    concept = key or str(raw_concept).strip()
                    group_key = (domain, concept)
                    summary = grouped.setdefault(
                        group_key,
                        EwokConceptSummary(domain=domain, concept=concept),
                    )
                    summary.eval_count += 1
                    summary.field_counts[field_name] += 1
                    summary.source_files.add(path.name)

    return sorted(grouped.values(), key=lambda item: (item.domain, item.concept))


def lexical_target_id(summary: EwokConceptSummary) -> str:
    return f"ewok_{slugify(summary.domain)}_{slugify(summary.concept)}"


def build_lexical_targets_payload(
    summaries: list[EwokConceptSummary],
    *,
    source_name: str = "ewok_fast",
) -> dict[str, Any]:
    """Build a v1 lexical target YAML payload for single-token concepts."""

    targets = []
    for summary in summaries:
        if not summary.is_supported:
            continue
        targets.append(
            {
                "id": lexical_target_id(summary),
                "label": summary.concept,
                "domain": summary.domain,
                "surfaces": [summary.concept],
                "lemmas": [summary.concept],
                "metadata": {
                    "source": source_name,
                    "eval_count": int(summary.eval_count),
                    "concept_fields": dict(sorted(summary.field_counts.items())),
                    "source_files": sorted(summary.source_files),
                },
            }
        )
    return {"version": 1, "targets": targets}


def build_domain_concepts_payload(summaries: list[EwokConceptSummary]) -> dict[str, Any]:
    """Build a domain concept YAML payload, including multiword concepts."""

    domains: dict[str, list[str]] = defaultdict(list)
    for summary in summaries:
        if is_placeholder_concept(summary.concept):
            continue
        domains[summary.domain].append(summary.concept)

    return {
        "version": 1,
        "domains": {
            domain: {"concepts": sorted(set(concepts))}
            for domain, concepts in sorted(domains.items())
        },
    }


def unsupported_rows(summaries: list[EwokConceptSummary]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        reason = summary.reason_unsupported
        if reason is None:
            continue
        rows.append(
            {
                "domain": summary.domain,
                "concept": summary.concept,
                "reason": reason,
                "eval_count": int(summary.eval_count),
                "concept_fields": " ".join(
                    f"{field}:{count}"
                    for field, count in sorted(summary.field_counts.items())
                ),
                "source_files": " ".join(sorted(summary.source_files)),
            }
        )
    return rows


def write_yaml(payload: dict[str, Any], path: str | Path) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def write_unsupported_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "concept",
        "reason",
        "eval_count",
        "concept_fields",
        "source_files",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_ewok_targets(
    *,
    ewok_dir: str | Path,
    lexical_targets_yaml: str | Path,
    domain_concepts_yaml: str | Path,
    unsupported_csv: str | Path,
    source_name: str = "ewok_fast",
) -> dict[str, Any]:
    """Export EWoK ConceptA/ConceptB into target/config files."""

    summaries = read_ewok_concepts(ewok_dir)
    lexical_payload = build_lexical_targets_payload(summaries, source_name=source_name)
    domain_payload = build_domain_concepts_payload(summaries)
    unsupported = unsupported_rows(summaries)

    write_yaml(lexical_payload, lexical_targets_yaml)
    write_yaml(domain_payload, domain_concepts_yaml)
    write_unsupported_csv(unsupported, unsupported_csv)

    return {
        "concepts_total": len(summaries),
        "lexical_targets": len(lexical_payload["targets"]),
        "unsupported_targets": len(unsupported),
        "domains": len(domain_payload["domains"]),
        "lexical_targets_yaml": str(lexical_targets_yaml),
        "domain_concepts_yaml": str(domain_concepts_yaml),
        "unsupported_csv": str(unsupported_csv),
    }


__all__ = [
    "EwokConceptSummary",
    "build_domain_concepts_payload",
    "build_lexical_targets_payload",
    "export_ewok_targets",
    "read_ewok_concepts",
    "unsupported_rows",
]
