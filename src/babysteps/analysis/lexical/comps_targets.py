"""Export COMPS concepts and property phrases into lexical-analysis targets."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from babysteps.analysis.lexical.targets import lexeme_tokens, normalize_lexeme


CONCEPT_FIELDS = ("acceptable_concept", "unacceptable_concept")
PLACEHOLDER_VALUES = {"", "-"}
_SLUG_RE = re.compile(r"[^a-z0-9]+")
SUBTASK_BY_FILE = {
    "comps_base": "base",
    "comps_wugs": "wugs",
    "comps_wugs_dist-before": "wugs_dist_before",
    "comps_wugs_dist-in-between": "wugs_dist_in_between",
}


@dataclass
class CompsConceptSummary:
    """Aggregated COMPS concept usage across acceptable/unacceptable roles."""

    concept: str
    eval_count: int = 0
    role_counts: Counter[str] = field(default_factory=Counter)
    subtask_counts: Counter[str] = field(default_factory=Counter)
    negative_sample_counts: Counter[str] = field(default_factory=Counter)
    distraction_counts: Counter[str] = field(default_factory=Counter)
    source_files: set[str] = field(default_factory=set)

    @property
    def reason_unsupported(self) -> str | None:
        if self.concept in PLACEHOLDER_VALUES:
            return "placeholder"
        if not lexeme_tokens(self.concept):
            return "empty"
        return None

    @property
    def is_supported(self) -> bool:
        return self.reason_unsupported is None


@dataclass
class CompsPropertySummary:
    """Aggregated COMPS property-phrase usage."""

    property_phrase: str
    eval_count: int = 0
    subtask_counts: Counter[str] = field(default_factory=Counter)
    negative_sample_counts: Counter[str] = field(default_factory=Counter)
    distraction_counts: Counter[str] = field(default_factory=Counter)
    source_files: set[str] = field(default_factory=set)

    @property
    def reason_unsupported(self) -> str | None:
        if self.property_phrase in PLACEHOLDER_VALUES:
            return "placeholder"
        if not lexeme_tokens(self.property_phrase):
            return "empty"
        return None

    @property
    def is_supported(self) -> bool:
        return self.reason_unsupported is None


def concept_key(value: Any) -> str:
    if value is None:
        return ""
    return normalize_lexeme(str(value).replace("_", " "))


def property_key(row: dict[str, Any]) -> str:
    value = row.get("property")
    if value is None:
        value = row.get("property_phrase", "")
    return normalize_lexeme(str(value).rstrip("."))


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", normalize_lexeme(value)).strip("_")
    return slug or "target"


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _metadata(
    *,
    source_name: str,
    eval_count: int,
    source_files: set[str],
    subtask_counts: Counter[str],
    negative_sample_counts: Counter[str],
    distraction_counts: Counter[str],
    role_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source_name,
        "eval_count": int(eval_count),
        "subtasks": _counter_payload(subtask_counts),
        "negative_sample_types": _counter_payload(negative_sample_counts),
        "distraction_types": _counter_payload(distraction_counts),
        "source_files": sorted(source_files),
    }
    if role_counts is not None:
        payload["concept_roles"] = _counter_payload(role_counts)
    return payload


def read_comps_targets(
    comps_dir: str | Path,
) -> tuple[list[CompsConceptSummary], list[CompsPropertySummary]]:
    """Read COMPS JSONL files and aggregate concept/property target summaries."""

    comps_dir = Path(comps_dir)
    if not comps_dir.exists():
        raise FileNotFoundError(f"Missing COMPS directory: {comps_dir}")

    paths = sorted(comps_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No COMPS JSONL files found in: {comps_dir}")

    concepts: dict[str, CompsConceptSummary] = {}
    properties: dict[str, CompsPropertySummary] = {}
    for path in paths:
        subtask = SUBTASK_BY_FILE.get(path.stem, path.stem)
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object at {path}:{line_number}.")
                negative_sample_type = str(row.get("negative_sample_type", ""))
                distraction_type = str(row.get("distraction_type", "none"))

                phrase = property_key(row)
                property_summary = properties.setdefault(
                    phrase,
                    CompsPropertySummary(property_phrase=phrase),
                )
                property_summary.eval_count += 1
                property_summary.subtask_counts[subtask] += 1
                property_summary.negative_sample_counts[negative_sample_type] += 1
                property_summary.distraction_counts[distraction_type] += 1
                property_summary.source_files.add(path.name)

                for role in CONCEPT_FIELDS:
                    concept = concept_key(row.get(role, ""))
                    summary = concepts.setdefault(
                        concept,
                        CompsConceptSummary(concept=concept),
                    )
                    summary.eval_count += 1
                    summary.role_counts[role] += 1
                    summary.subtask_counts[subtask] += 1
                    summary.negative_sample_counts[negative_sample_type] += 1
                    summary.distraction_counts[distraction_type] += 1
                    summary.source_files.add(path.name)

    return (
        sorted(concepts.values(), key=lambda item: item.concept),
        sorted(properties.values(), key=lambda item: item.property_phrase),
    )


def concept_target_id(summary: CompsConceptSummary) -> str:
    return f"comps_concept_{slugify(summary.concept)}"


def property_target_id(summary: CompsPropertySummary) -> str:
    return f"comps_property_{slugify(summary.property_phrase)}"


def build_concept_targets_payload(
    summaries: list[CompsConceptSummary],
    *,
    source_name: str = "comps",
) -> dict[str, Any]:
    targets = []
    for summary in summaries:
        if not summary.is_supported:
            continue
        target_type = "phrase" if len(lexeme_tokens(summary.concept)) > 1 else "word"
        targets.append(
            {
                "id": concept_target_id(summary),
                "label": summary.concept,
                "target_type": target_type,
                "surfaces": [summary.concept],
                "lemmas": [summary.concept],
                "metadata": _metadata(
                    source_name=source_name,
                    eval_count=summary.eval_count,
                    role_counts=summary.role_counts,
                    subtask_counts=summary.subtask_counts,
                    negative_sample_counts=summary.negative_sample_counts,
                    distraction_counts=summary.distraction_counts,
                    source_files=summary.source_files,
                ),
            }
        )
    return {"version": 2, "targets": targets}


def build_property_targets_payload(
    summaries: list[CompsPropertySummary],
    *,
    source_name: str = "comps",
) -> dict[str, Any]:
    targets = []
    for summary in summaries:
        if not summary.is_supported:
            continue
        targets.append(
            {
                "id": property_target_id(summary),
                "label": summary.property_phrase,
                "target_type": "phrase",
                "surfaces": [summary.property_phrase],
                "lemmas": [summary.property_phrase],
                "metadata": _metadata(
                    source_name=source_name,
                    eval_count=summary.eval_count,
                    subtask_counts=summary.subtask_counts,
                    negative_sample_counts=summary.negative_sample_counts,
                    distraction_counts=summary.distraction_counts,
                    source_files=summary.source_files,
                ),
            }
        )
    return {"version": 2, "targets": targets}


def unsupported_rows(
    *,
    concept_summaries: list[CompsConceptSummary],
    property_summaries: list[CompsPropertySummary],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in concept_summaries:
        reason = summary.reason_unsupported
        if reason is None:
            continue
        rows.append(
            {
                "target_kind": "concept",
                "target": summary.concept,
                "reason": reason,
                "eval_count": int(summary.eval_count),
                "source_files": " ".join(sorted(summary.source_files)),
            }
        )
    for summary in property_summaries:
        reason = summary.reason_unsupported
        if reason is None:
            continue
        rows.append(
            {
                "target_kind": "property_phrase",
                "target": summary.property_phrase,
                "reason": reason,
                "eval_count": int(summary.eval_count),
                "source_files": " ".join(sorted(summary.source_files)),
            }
        )
    return rows


def write_yaml(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # JSON is valid YAML and keeps this generated artifact independent of PyYAML.
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_unsupported_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["target_kind", "target", "reason", "eval_count", "source_files"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_comps_targets(
    *,
    comps_dir: str | Path,
    concept_targets_yaml: str | Path,
    property_targets_yaml: str | Path,
    unsupported_csv: str | Path,
    source_name: str = "comps",
) -> dict[str, Any]:
    """Export COMPS concepts and properties into target files."""

    concept_summaries, property_summaries = read_comps_targets(comps_dir)
    concept_payload = build_concept_targets_payload(
        concept_summaries,
        source_name=source_name,
    )
    property_payload = build_property_targets_payload(
        property_summaries,
        source_name=source_name,
    )
    unsupported = unsupported_rows(
        concept_summaries=concept_summaries,
        property_summaries=property_summaries,
    )

    write_yaml(concept_payload, concept_targets_yaml)
    write_yaml(property_payload, property_targets_yaml)
    write_unsupported_csv(unsupported, unsupported_csv)

    return {
        "concepts_total": len(concept_summaries),
        "concept_targets": len(concept_payload["targets"]),
        "properties_total": len(property_summaries),
        "property_targets": len(property_payload["targets"]),
        "unsupported_targets": len(unsupported),
        "concept_targets_yaml": str(concept_targets_yaml),
        "property_targets_yaml": str(property_targets_yaml),
        "unsupported_csv": str(unsupported_csv),
    }


__all__ = [
    "CompsConceptSummary",
    "CompsPropertySummary",
    "build_concept_targets_payload",
    "build_property_targets_payload",
    "export_comps_targets",
    "read_comps_targets",
    "unsupported_rows",
]
