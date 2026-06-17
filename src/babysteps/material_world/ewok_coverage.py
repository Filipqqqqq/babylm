"""Coverage audits against EWoK material-domain evaluation files.

The prose audit checks whether generated examples are diverse and readable.
This module checks a different question: did a generated corpus cover the
specific concepts and lexical anchors used by an EWoK evaluation split?
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MATERIAL_DOMAINS = ("material-dynamics", "material-properties")

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class GeneratedCorpus:
    """Generated text plus optional metadata strings for coverage matching."""

    path: Path
    texts: list[str]
    metadata_texts: list[str]
    row_count: int
    kind: str


def normalize_term(value: Any) -> str:
    """Normalize an EWoK or metadata term for counting."""

    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip(" .,:;!?\"'()[]{}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _metadata_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_metadata_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(_metadata_strings(item))
        return strings
    return [str(value)]


def load_generated_corpus(path: str | Path, *, text_field: str = "text") -> GeneratedCorpus:
    """Load generated text from a JSONL or plain text file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing generated corpus: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        texts = [str(row.get(text_field, "")).strip() for row in rows]
        metadata_texts = [
            " ".join(_metadata_strings(row.get("metadata", {})))
            for row in rows
        ]
        return GeneratedCorpus(
            path=path,
            texts=[text for text in texts if text],
            metadata_texts=[text for text in metadata_texts if text.strip()],
            row_count=len(rows),
            kind="jsonl",
        )

    text = path.read_text(encoding="utf-8")
    documents = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return GeneratedCorpus(
        path=path,
        texts=documents,
        metadata_texts=[],
        row_count=len(documents),
        kind="text",
    )


def _add_counter_term(counter: Counter[str], value: Any, amount: int = 1) -> None:
    term = normalize_term(value)
    if term:
        counter[term] += amount


def _extract_context_anchors(context: str) -> list[str]:
    """Extract material/object anchors from common EWoK context templates."""

    context = normalize_term(context)
    anchors: list[str] = []
    patterns = [
        r"\bsees something that is ([a-z0-9][a-z0-9 '\-]*?)(?:\.|$)",
        r"\bsee something that is ([a-z0-9][a-z0-9 '\-]*?)(?:\.|$)",
        r"\bsees the ([a-z0-9][a-z0-9 '\-]*?)(?:\.|$)",
        r"\bsee the ([a-z0-9][a-z0-9 '\-]*?)(?:\.|$)",
        r"\bmade of ([a-z0-9][a-z0-9 '\-]*?)(?:\.|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, context):
            anchor = normalize_term(match.group(1))
            if anchor and anchor not in {"a", "an", "the", "something"}:
                anchors.append(anchor)
    return anchors


def load_ewok_targets(
    ewok_dir: str | Path,
    *,
    domains: Sequence[str] = DEFAULT_MATERIAL_DOMAINS,
) -> dict[str, dict[str, Any]]:
    """Load EWoK JSONL files and summarize domain concepts and anchors."""

    ewok_dir = Path(ewok_dir)
    if not ewok_dir.exists():
        raise FileNotFoundError(f"Missing EWoK directory: {ewok_dir}")

    targets: dict[str, dict[str, Any]] = {}
    for domain in domains:
        path = ewok_dir / f"{domain}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing EWoK domain file: {path}")
        rows = read_jsonl(path)
        concept_counts: Counter[str] = Counter()
        concept_pair_counts: Counter[str] = Counter()
        anchor_counts: Counter[str] = Counter()
        context_type_counts: Counter[str] = Counter()
        context_diff_counts: Counter[str] = Counter()
        target_diff_counts: Counter[str] = Counter()
        anchor_examples: dict[str, list[str]] = defaultdict(list)

        for row in rows:
            concept_a = normalize_term(row.get("ConceptA", ""))
            concept_b = normalize_term(row.get("ConceptB", ""))
            for concept in (concept_a, concept_b):
                if concept:
                    concept_counts[concept] += 1
            if concept_a or concept_b:
                pair = " | ".join(term for term in (concept_a, concept_b) if term)
                concept_pair_counts[pair] += 1

            _add_counter_term(context_type_counts, row.get("ContextType", ""))
            _add_counter_term(context_diff_counts, row.get("ContextDiff", ""))
            _add_counter_term(target_diff_counts, row.get("TargetDiff", ""))

            for field in ("Context1", "Context2"):
                context = str(row.get(field, ""))
                for anchor in _extract_context_anchors(context):
                    anchor_counts[anchor] += 1
                    if len(anchor_examples[anchor]) < 3:
                        anchor_examples[anchor].append(context)

        targets[domain] = {
            "source_path": str(path),
            "item_count": len(rows),
            "concept_counts": dict(sorted(concept_counts.items())),
            "concept_pair_counts": dict(sorted(concept_pair_counts.items())),
            "anchor_counts": dict(sorted(anchor_counts.items())),
            "context_type_counts": dict(sorted(context_type_counts.items())),
            "context_diff_counts": dict(sorted(context_diff_counts.items())),
            "target_diff_counts": dict(sorted(target_diff_counts.items())),
            "anchor_examples": dict(sorted(anchor_examples.items())),
        }

    return targets


def _term_variants(term: str) -> list[str]:
    term = normalize_term(term)
    if not term:
        return []
    variants = {term}
    if "-" in term:
        variants.add(term.replace("-", " "))
    if " " in term:
        variants.add(term.replace(" ", "-"))

    words = term.split()
    if words:
        last = words[-1]
        plural_last = None
        if last.endswith("y") and len(last) > 1:
            plural_last = last[:-1] + "ies"
        elif last.endswith("s"):
            plural_last = last + "es"
        else:
            plural_last = last + "s"
        plural_words = [*words[:-1], plural_last]
        variants.add(" ".join(plural_words))
    if term == "gas":
        variants.add("gases")
    if term == "play-doh":
        variants.add("play dough")
    if term == "jelly":
        variants.add("jello")

    return sorted(variants, key=len, reverse=True)


def _pattern_for_variant(variant: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in re.split(r"[\s\-]+", variant) if part]
    body = r"[\s\-]+".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def count_term_mentions(text: str, term: str) -> int:
    """Count term mentions with simple word-boundary and plural handling."""

    if not term:
        return 0
    spans = set()
    for variant in _term_variants(term):
        for match in _pattern_for_variant(variant).finditer(text):
            spans.add(match.span())
    return len(spans)


def _counts_for_terms(terms: Iterable[str], texts: Sequence[str]) -> dict[str, int]:
    joined = "\n".join(texts).lower()
    return {term: count_term_mentions(joined, term) for term in terms}


def _coverage_entries(
    target_counts: dict[str, int],
    text_counts: dict[str, int],
    metadata_counts: dict[str, int],
    *,
    min_mentions: int,
) -> dict[str, dict[str, Any]]:
    entries = {}
    for term, eval_count in sorted(target_counts.items()):
        text_count = int(text_counts.get(term, 0))
        metadata_count = int(metadata_counts.get(term, 0))
        total_generated_count = text_count + metadata_count
        entries[term] = {
            "eval_count": int(eval_count),
            "generated_text_count": text_count,
            "generated_metadata_count": metadata_count,
            "generated_total_count": total_generated_count,
            "covered": total_generated_count >= min_mentions,
        }
    return entries


def _coverage_summary(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    covered_terms = [term for term, entry in entries.items() if entry["covered"]]
    missing_terms = [term for term, entry in entries.items() if not entry["covered"]]
    return {
        "target_count": total,
        "covered_count": len(covered_terms),
        "coverage_rate": (len(covered_terms) / total) if total else None,
        "covered": covered_terms,
        "missing": missing_terms,
    }


def build_coverage_audit(
    *,
    ewok_targets: dict[str, dict[str, Any]],
    generated: GeneratedCorpus | None = None,
    min_anchor_mentions: int = 1,
    min_concept_mentions: int = 1,
) -> dict[str, Any]:
    """Build a target-only or posthoc EWoK coverage audit report."""

    report: dict[str, Any] = {
        "version": "ewok_material_coverage_audit_v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated": None,
        "domains": {},
        "summary": {},
    }

    if generated is not None:
        report["generated"] = {
            "path": str(generated.path),
            "kind": generated.kind,
            "row_count": generated.row_count,
            "text_documents": len(generated.texts),
            "metadata_documents": len(generated.metadata_texts),
        }

    summary_domains = {}
    for domain, target in ewok_targets.items():
        concept_counts = {term: int(count) for term, count in target["concept_counts"].items()}
        anchor_counts = {term: int(count) for term, count in target["anchor_counts"].items()}

        if generated is None:
            concept_entries = _coverage_entries(
                concept_counts,
                {},
                {},
                min_mentions=min_concept_mentions,
            )
            anchor_entries = _coverage_entries(
                anchor_counts,
                {},
                {},
                min_mentions=min_anchor_mentions,
            )
        else:
            concept_text_counts = _counts_for_terms(concept_counts, generated.texts)
            concept_metadata_counts = _counts_for_terms(concept_counts, generated.metadata_texts)
            anchor_text_counts = _counts_for_terms(anchor_counts, generated.texts)
            anchor_metadata_counts = _counts_for_terms(anchor_counts, generated.metadata_texts)
            concept_entries = _coverage_entries(
                concept_counts,
                concept_text_counts,
                concept_metadata_counts,
                min_mentions=min_concept_mentions,
            )
            anchor_entries = _coverage_entries(
                anchor_counts,
                anchor_text_counts,
                anchor_metadata_counts,
                min_mentions=min_anchor_mentions,
            )

        concept_summary = _coverage_summary(concept_entries)
        anchor_summary = _coverage_summary(anchor_entries)
        warnings = []
        if generated is not None and anchor_summary["missing"]:
            warnings.append(
                f"{domain} is missing {len(anchor_summary['missing'])} lexical anchors."
            )
        if generated is not None and concept_summary["missing"]:
            warnings.append(
                f"{domain} is missing {len(concept_summary['missing'])} concepts."
            )

        report["domains"][domain] = {
            "source_path": target["source_path"],
            "item_count": target["item_count"],
            "context_type_counts": target["context_type_counts"],
            "context_diff_counts": target["context_diff_counts"],
            "target_diff_counts": target["target_diff_counts"],
            "concept_pairs": target["concept_pair_counts"],
            "concepts": {
                "summary": concept_summary,
                "terms": concept_entries,
            },
            "lexical_anchors": {
                "summary": anchor_summary,
                "terms": anchor_entries,
                "examples": target["anchor_examples"],
            },
            "warnings": warnings,
        }
        summary_domains[domain] = {
            "concepts_covered": (
                f"{concept_summary['covered_count']}/{concept_summary['target_count']}"
            ),
            "lexical_anchors_covered": (
                f"{anchor_summary['covered_count']}/{anchor_summary['target_count']}"
            ),
            "missing_concepts": concept_summary["missing"],
            "missing_lexical_anchors": anchor_summary["missing"],
        }

    report["summary"] = {
        "domains": summary_domains,
        "warnings": [
            warning
            for domain_report in report["domains"].values()
            for warning in domain_report["warnings"]
        ],
    }
    return report


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def audit_ewok_coverage(
    *,
    ewok_dir: str | Path,
    report_json: str | Path,
    generated_path: str | Path | None = None,
    domains: Sequence[str] = DEFAULT_MATERIAL_DOMAINS,
    text_field: str = "text",
    min_anchor_mentions: int = 1,
    min_concept_mentions: int = 1,
) -> dict[str, Any]:
    """Load targets, optionally load generated text, write a coverage report."""

    targets = load_ewok_targets(ewok_dir, domains=domains)
    generated = (
        load_generated_corpus(generated_path, text_field=text_field)
        if generated_path is not None
        else None
    )
    report = build_coverage_audit(
        ewok_targets=targets,
        generated=generated,
        min_anchor_mentions=min_anchor_mentions,
        min_concept_mentions=min_concept_mentions,
    )
    write_json(report, report_json)
    return report
