"""Domain-level frequency tables and plots for lexical concepts."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from babysteps.analysis.lexical.targets import normalize_lexeme


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class DomainConcept:
    """One concept assigned to a reporting domain."""

    domain: str
    concept: str
    key: str
    is_multiword: bool


def concept_key(value: Any) -> str:
    """Normalize a concept for matching coverage rows and detecting phrases."""

    text = normalize_lexeme(value).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_multiword_concept(value: str) -> bool:
    """Return whether v1 frequency plots should leave this concept uncounted."""

    return bool(re.search(r"\s+", value.strip()))


def slugify(value: str) -> str:
    """Build a stable, filesystem-friendly slug."""

    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "domain"


def load_domain_concepts(path: str | Path) -> list[DomainConcept]:
    """Load domain concepts from YAML.

    Supported shapes:

    ``domains`` mapping:
        domains:
          material:
            concepts: [soft, hard, ice cream]

    ``targets`` list:
        targets:
          - id: soft
            domain: material
    """

    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Domain concept YAML must contain a mapping: {path}")

    concepts: list[DomainConcept] = []
    if "domains" in payload:
        domains = payload["domains"]
        if not isinstance(domains, dict):
            raise TypeError("'domains' must be a mapping of domain name to concepts.")
        for domain, raw_domain in domains.items():
            if isinstance(raw_domain, dict):
                raw_concepts = raw_domain.get("concepts", [])
            else:
                raw_concepts = raw_domain
            if not isinstance(raw_concepts, list):
                raise TypeError(f"Domain {domain!r} concepts must be a list.")
            for raw_concept in raw_concepts:
                concepts.append(_domain_concept(str(domain), raw_concept))

    if "targets" in payload:
        raw_targets = payload["targets"]
        if not isinstance(raw_targets, list):
            raise TypeError("'targets' must be a list when provided.")
        for index, raw_target in enumerate(raw_targets):
            if not isinstance(raw_target, dict):
                raise TypeError(f"Target at index {index} must be a mapping.")
            domain = str(raw_target.get("domain", "default"))
            raw_concept = raw_target.get("label", raw_target.get("id"))
            if raw_concept is None:
                raise ValueError(f"Target at index {index} is missing label/id.")
            concepts.append(_domain_concept(domain, raw_concept))

    if not concepts:
        raise ValueError("Domain concept YAML must define 'domains' or 'targets'.")
    return concepts


def _domain_concept(domain: str, raw_concept: Any) -> DomainConcept:
    concept = str(raw_concept).strip()
    key = concept_key(concept)
    if not concept or not key:
        raise ValueError(f"Empty concept in domain {domain!r}.")
    return DomainConcept(
        domain=str(domain).strip() or "default",
        concept=concept,
        key=key,
        is_multiword=is_multiword_concept(key),
    )


def load_frequency_lookup(
    target_coverage_csv: str | Path,
    *,
    frequency_column: str = "total_mentions",
) -> dict[str, float]:
    """Read target coverage rows into a normalized concept-frequency lookup."""

    lookup: dict[str, float] = {}
    with Path(target_coverage_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_value = row.get(frequency_column, "")
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = math.nan

            keys = [
                concept_key(row.get("target_id", "")),
                concept_key(row.get("label", "")),
            ]
            for key in keys:
                if key:
                    lookup[key] = value
    return lookup


def build_domain_frequency_rows(
    concepts: list[DomainConcept],
    frequency_lookup: dict[str, float],
    *,
    sort_by: str = "frequency",
) -> list[dict[str, Any]]:
    """Build plot-ready rows, leaving multiword concepts as NaN for v1."""

    rows: list[dict[str, Any]] = []
    for item in concepts:
        if item.is_multiword:
            frequency = math.nan
            status = "multiword_unsupported"
        elif item.key not in frequency_lookup:
            frequency = math.nan
            status = "missing_from_target_coverage"
        else:
            frequency = frequency_lookup[item.key]
            status = "counted"

        rows.append(
            {
                "domain": item.domain,
                "concept": item.concept,
                "concept_key": item.key,
                "frequency": frequency,
                "status": status,
            }
        )
    return sort_domain_frequency_rows(rows, sort_by=sort_by)


def sort_domain_frequency_rows(
    rows: list[dict[str, Any]],
    *,
    sort_by: str = "frequency",
) -> list[dict[str, Any]]:
    """Sort rows within each domain for stable CSVs and plots."""

    sort_by = sort_by.strip().lower()
    if sort_by not in {"frequency", "input"}:
        raise ValueError("sort_by must be 'frequency' or 'input'.")
    if sort_by == "input":
        return list(rows)

    indexed_rows = list(enumerate(rows))
    indexed_rows.sort(
        key=lambda item: (
            str(item[1]["domain"]),
            _frequency_sort_key(item[1]["frequency"]),
            item[0],
        )
    )
    return [row for _index, row in indexed_rows]


def _frequency_sort_key(value: Any) -> tuple[int, float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if math.isnan(number):
        return (1, 0.0)
    return (0, -number)


def write_domain_frequency_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["domain", "concept", "concept_key", "frequency", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            if isinstance(out_row["frequency"], float) and math.isnan(out_row["frequency"]):
                out_row["frequency"] = "NaN"
            writer.writerow(out_row)


def plot_domain_frequency_rows(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    image_format: str = "png",
    log_y: bool = False,
) -> list[Path]:
    """Write one concept-frequency plot per domain."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    domains = sorted({str(row["domain"]) for row in rows})
    for domain in domains:
        domain_rows = [row for row in rows if row["domain"] == domain]
        concepts = [str(row["concept"]) for row in domain_rows]
        frequencies = [float(row["frequency"]) for row in domain_rows]
        is_nan = [math.isnan(value) for value in frequencies]
        plot_values = [0.0 if missing else value for missing, value in zip(is_nan, frequencies)]

        width = max(8.0, min(24.0, 0.45 * len(concepts) + 4.0))
        fig, ax = plt.subplots(figsize=(width, 5.5))
        colors = ["#c7c7c7" if missing else "#4c78a8" for missing in is_nan]
        bars = ax.bar(range(len(concepts)), plot_values, color=colors)
        for bar, missing in zip(bars, is_nan):
            if missing:
                bar.set_hatch("//")
                bar.set_alpha(0.55)

        numeric_values = [value for value, missing in zip(frequencies, is_nan) if not missing]
        y_max = max(numeric_values) if numeric_values else 1.0
        ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1.0)
        if log_y and any(value > 0 for value in numeric_values):
            ax.set_yscale("log")
            ax.set_ylim(max(0.5, min(value for value in numeric_values if value > 0) * 0.8), y_max * 1.4)

        for index, missing in enumerate(is_nan):
            if missing:
                ax.text(
                    index,
                    ax.get_ylim()[1] * 0.03,
                    "NaN",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    color="#555555",
                    fontsize=8,
                )

        ax.set_title(f"{domain} concept frequencies")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Concept")
        ax.set_xticks(range(len(concepts)))
        ax.set_xticklabels(concepts, rotation=45, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()

        path = output_dir / f"{slugify(domain)}_concept_frequencies.{image_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def build_domain_frequency_report(
    *,
    target_coverage_csv: str | Path,
    concepts_yaml: str | Path,
    output_dir: str | Path,
    frequency_column: str = "total_mentions",
    image_format: str = "png",
    log_y: bool = False,
    make_plots: bool = True,
    sort_by: str = "frequency",
) -> dict[str, Any]:
    """Build CSV data and optional per-domain plots."""

    concepts = load_domain_concepts(concepts_yaml)
    lookup = load_frequency_lookup(
        target_coverage_csv,
        frequency_column=frequency_column,
    )
    rows = build_domain_frequency_rows(concepts, lookup, sort_by=sort_by)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "domain_concept_frequencies.csv"
    write_domain_frequency_csv(rows, csv_path)
    plot_paths = (
        plot_domain_frequency_rows(
            rows,
            output_dir,
            image_format=image_format,
            log_y=log_y,
        )
        if make_plots
        else []
    )
    return {
        "csv": csv_path,
        "plots": plot_paths,
        "concept_count": len(rows),
        "domain_count": len({row["domain"] for row in rows}),
    }


__all__ = [
    "DomainConcept",
    "build_domain_frequency_report",
    "build_domain_frequency_rows",
    "concept_key",
    "load_domain_concepts",
    "load_frequency_lookup",
    "plot_domain_frequency_rows",
    "sort_domain_frequency_rows",
    "write_domain_frequency_csv",
]
