"""Build a reusable COMPS concept-property fact and template bank."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from babysteps.material_world.fireworks_client import FireworksChatClient, load_api_key
from babysteps.paths import resolve_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = (
    "data/concept_curriculum/comps_concept_properties_clean_no_neighbors_full521_qwen"
)
DEFAULT_BY_CONCEPT = "documents_by_concept_less_repetitive_reasoning_low_preserve_details.jsonl"
DEFAULT_PARAGRAPHS = "documents_less_repetitive_reasoning_low_preserve_details.jsonl"
DEFAULT_MODEL_ID = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_SEED = 42
DEFAULT_MIN_FACTS = 8
DEFAULT_MAX_FACTS = 12
DEFAULT_EXTRACTION_MAX_TOKENS = 3072
DEFAULT_TEMPLATE_MAX_TOKENS = 4096
DEFAULT_EXTRACTION_TEMPERATURE = 0.2
DEFAULT_TEMPLATE_TEMPERATURE = 0.4
PROMPT_ONLY = "prompt_only"
DETERMINISTIC = "deterministic"
FIREWORKS = "fireworks"
RENDERER_TYPES = {PROMPT_ONLY, DETERMINISTIC, FIREWORKS}

PARTS_STYLE = "parts_and_properties"
SPECIALIZED_STYLES = (
    "operation_and_components",
    "life_cycle_and_movement",
    "body_and_signals",
    "ecology_and_growth",
    "ingredients_and_serving",
    "forms_and_uses",
)
RANDOM_PREFERRED_STYLES = ("core_profile", "contexts_and_interactions")
NATURAL_CONTRASTS = "natural_contrasts"

SPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
GENERIC_PATTERNS = (
    re.compile(r"\bis (?:common|useful|important|popular|familiar|well known)\b", re.I),
    re.compile(r"\bcan be found\b", re.I),
    re.compile(r"\bhas (?:parts|features|properties)\b", re.I),
    re.compile(r"\bis used (?:by people|in many ways|for many things)\b", re.I),
)
TAXONOMY_ONLY_PATTERN = re.compile(
    r"^\s*(?:it\s+)?is\s+(?:a|an|the|one|part|member|kind|type|form|variety)\b",
    re.I,
)
WUGS_TEMPLATE_PATTERN = re.compile(
    r"\b(?:wug|fep|dax|blicket)\b|\btherefore\b|^\s*A\s+\{handle\}\s+is\s+an?\b",
    re.I,
)


@dataclass(frozen=True)
class SourceParagraph:
    concept_id: str
    concept: str
    concept_types: tuple[str, ...]
    paragraph_id: str
    style_id: str
    text: str
    text_source: str
    selection_round: str
    selection_reason: str
    source_index: int


@dataclass(frozen=True)
class PromptRecord:
    id: str
    concept_id: str
    concept: str
    stage: str
    selection_round: str
    paragraph_ids: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class FactCandidate:
    id: str
    concept_id: str
    concept: str
    extraction_round: str
    fact: str
    evidence: str
    source_paragraph_id: str
    fact_type: str
    handle_safe: bool
    specificity: str
    raw_index: int


@dataclass(frozen=True)
class TemplateCandidate:
    id: str
    concept_id: str
    concept: str
    fact_id: str
    fact: str
    template: str
    template_type: str
    raw_index: int


@dataclass(frozen=True)
class PropertyBankConfig:
    input_dir: Path
    output_dir: Path
    by_concept_filename: str = DEFAULT_BY_CONCEPT
    paragraph_filename: str = DEFAULT_PARAGRAPHS
    seed: int = DEFAULT_SEED
    renderer: str = PROMPT_ONLY
    model_id: str = DEFAULT_MODEL_ID
    concept_limit: int | None = None
    concept_ids: tuple[str, ...] = ()
    min_facts_per_concept: int = DEFAULT_MIN_FACTS
    max_facts_per_concept: int = DEFAULT_MAX_FACTS
    extraction_temperature: float = DEFAULT_EXTRACTION_TEMPERATURE
    template_temperature: float = DEFAULT_TEMPLATE_TEMPERATURE
    extraction_max_tokens: int = DEFAULT_EXTRACTION_MAX_TOKENS
    template_max_tokens: int = DEFAULT_TEMPLATE_MAX_TOKENS
    reasoning_effort: str = "none"
    reasoning_history: str = "disabled"
    timeout_seconds: int = 600
    max_request_retries: int = 4
    force: bool = False


@dataclass(frozen=True)
class PropertyBankResult:
    outputs: dict[str, Path]
    manifest: dict[str, Any]
    audit_report: dict[str, Any]


class PropertyBankRenderer(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_space(text: str) -> str:
    return SPACE_PATTERN.sub(" ", str(text).strip())


def normalize_key(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(str(text).lower()))


def strip_response_fence(raw: str) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        text = FENCE_PATTERN.sub("", text).strip()
    return text


def parse_json_object(raw: str) -> dict[str, Any]:
    text = strip_response_fence(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("renderer response must be a JSON object")
    return payload


def _plural_variants(word: str) -> set[str]:
    variants = {word}
    if not word:
        return variants
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        variants.add(f"{word[:-1]}ies")
    elif word.endswith(("s", "x", "ch", "sh")):
        variants.add(f"{word}es")
    else:
        variants.add(f"{word}s")
    return variants


def concept_leak_variants(concept: str) -> tuple[str, ...]:
    concept_text = normalize_space(concept.replace("_", " ")).lower()
    if not concept_text:
        return ()
    parts = concept_text.split()
    variants = {concept_text}
    for plural_last in _plural_variants(parts[-1]):
        variants.add(" ".join([*parts[:-1], plural_last]))
    return tuple(sorted(variants, key=len, reverse=True))


def concept_leaks(text: str, concept: str) -> bool:
    text_lower = str(text).lower()
    for variant in concept_leak_variants(concept):
        escaped = r"\s+".join(re.escape(part) for part in variant.split())
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text_lower) is not None:
            return True
    return False


def evidence_supported(evidence: str, paragraphs: dict[str, SourceParagraph]) -> bool:
    paragraph_texts = [normalize_space(paragraph.text) for paragraph in paragraphs.values()]
    needle = normalize_space(evidence)
    if not needle:
        return False
    return any(needle in text for text in paragraph_texts)


def sentence_split(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalize_space(text)) if part.strip()]


def source_paragraph_to_row(paragraph: SourceParagraph) -> dict[str, Any]:
    return {
        "concept": paragraph.concept,
        "concept_id": paragraph.concept_id,
        "concept_types": list(paragraph.concept_types),
        "id": f"{paragraph.concept_id}__{paragraph.paragraph_id}__{paragraph.selection_round}",
        "paragraph_id": paragraph.paragraph_id,
        "selection_reason": paragraph.selection_reason,
        "selection_round": paragraph.selection_round,
        "source_index": paragraph.source_index,
        "style_id": paragraph.style_id,
        "text": paragraph.text,
        "text_source": paragraph.text_source,
    }


def prompt_record_to_row(record: PromptRecord) -> dict[str, Any]:
    row = asdict(record)
    row["paragraph_ids"] = list(record.paragraph_ids)
    return row


def fact_to_row(fact: FactCandidate) -> dict[str, Any]:
    return asdict(fact)


def template_to_row(template: TemplateCandidate) -> dict[str, Any]:
    return asdict(template)


def _paragraph_rows_for_concept(
    concept_row: dict[str, Any],
    paragraph_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for index, paragraph in enumerate(concept_row.get("paragraphs") or []):
        paragraph_id = str(paragraph.get("id") or paragraph.get("prompt_id") or "").strip()
        if not paragraph_id:
            continue
        source = paragraph_lookup.get(paragraph_id, {})
        source_text = normalize_space(source.get("source_text") or "")
        source_kind = "source_text"
        if not source_text:
            source_text = normalize_space(source.get("text") or "")
            source_kind = "source_text_fallback_postprocessed"
        if not source_text:
            source_text = normalize_space(paragraph.get("text") or "")
            source_kind = "concept_export_text"
        rows.append(
            {
                "id": paragraph_id,
                "style_id": str(paragraph.get("style_id") or source.get("style_id") or "").strip(),
                "text": source_text,
                "text_source": source_kind,
                "index": int(paragraph.get("index", index)),
            }
        )
    return rows


def concept_types_from_row(concept_row: dict[str, Any]) -> tuple[str, ...]:
    values = concept_row.get("concept_types")
    if isinstance(values, list) and values:
        return tuple(str(item) for item in values)
    value = str(concept_row.get("concept_type") or "").strip()
    return (value,) if value else ()


def _choose_first_by_style(rows: list[dict[str, Any]], styles: tuple[str, ...], used: set[str]) -> dict[str, Any] | None:
    for style in styles:
        for row in rows:
            if row["id"] not in used and row.get("style_id") == style:
                return row
    return None


def _choose_seeded(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any] | None:
    if not rows:
        return None
    return rng.choice(rows)


def select_initial_paragraphs(
    *,
    concept_row: dict[str, Any],
    paragraph_lookup: dict[str, dict[str, Any]],
    rng: random.Random,
) -> list[SourceParagraph]:
    source_rows = _paragraph_rows_for_concept(concept_row, paragraph_lookup)
    concept_id = str(concept_row["concept_id"])
    concept = str(concept_row.get("concept") or concept_id.replace("_", " "))
    concept_types = concept_types_from_row(concept_row)
    selected: list[tuple[dict[str, Any], str]] = []
    used: set[str] = set()

    parts = _choose_first_by_style(source_rows, (PARTS_STYLE,), used)
    if parts is not None:
        selected.append((parts, "priority_parts_and_properties"))
        used.add(parts["id"])

    specialized = _choose_first_by_style(source_rows, SPECIALIZED_STYLES, used)
    if specialized is not None:
        selected.append((specialized, "priority_specialized_mechanism"))
        used.add(specialized["id"])

    remaining = [row for row in source_rows if row["id"] not in used]
    preferred = [row for row in remaining if row.get("style_id") in RANDOM_PREFERRED_STYLES]
    non_contrast = [row for row in remaining if row.get("style_id") != NATURAL_CONTRASTS]
    random_pick = _choose_seeded(preferred or non_contrast or remaining, rng)
    if random_pick is not None:
        selected.append((random_pick, "seeded_random_remaining"))
        used.add(random_pick["id"])

    while len(selected) < 3:
        remaining = [row for row in source_rows if row["id"] not in used]
        next_pick = _choose_seeded(remaining, rng)
        if next_pick is None:
            break
        selected.append((next_pick, "seeded_fill_to_three"))
        used.add(next_pick["id"])

    paragraphs = []
    for order, (row, reason) in enumerate(selected):
        paragraphs.append(
            SourceParagraph(
                concept_id=concept_id,
                concept=concept,
                concept_types=concept_types,
                paragraph_id=str(row["id"]),
                style_id=str(row.get("style_id") or ""),
                text=normalize_space(row.get("text") or ""),
                text_source=str(row.get("text_source") or ""),
                selection_round="initial",
                selection_reason=reason,
                source_index=order,
            )
        )
    return paragraphs


def select_fill_in_paragraphs(
    *,
    concept_row: dict[str, Any],
    paragraph_lookup: dict[str, dict[str, Any]],
    already_selected_ids: set[str],
    rng: random.Random,
    count: int = 2,
) -> list[SourceParagraph]:
    source_rows = _paragraph_rows_for_concept(concept_row, paragraph_lookup)
    remaining = [row for row in source_rows if row["id"] not in already_selected_ids]
    if not remaining:
        return []
    preferred = [row for row in remaining if row.get("style_id") != NATURAL_CONTRASTS]
    pool = preferred or remaining
    picks = rng.sample(pool, k=min(int(count), len(pool)))
    concept_id = str(concept_row["concept_id"])
    concept = str(concept_row.get("concept") or concept_id.replace("_", " "))
    concept_types = concept_types_from_row(concept_row)
    return [
        SourceParagraph(
            concept_id=concept_id,
            concept=concept,
            concept_types=concept_types,
            paragraph_id=str(row["id"]),
            style_id=str(row.get("style_id") or ""),
            text=normalize_space(row.get("text") or ""),
            text_source=str(row.get("text_source") or ""),
            selection_round="fill_in",
            selection_reason="fill_in_for_low_fact_count",
            source_index=index,
        )
        for index, row in enumerate(picks)
    ]


def build_extraction_prompt(
    *,
    concept_id: str,
    concept: str,
    concept_types: tuple[str, ...],
    paragraphs: list[SourceParagraph],
    disable_thinking: bool = True,
) -> str:
    type_text = ", ".join(concept_types) if concept_types else "unknown"
    paragraph_blocks = []
    for paragraph in paragraphs:
        paragraph_blocks.append(
            "\n".join(
                [
                    f"Paragraph ID: {paragraph.paragraph_id}",
                    f"Style: {paragraph.style_id}",
                    f"Text: {paragraph.text}",
                ]
            )
        )
    prompt = f"""You are building a property bank for entity-binding language-model training.

Concept ID: {concept_id}
Concept: {concept}
Concept types: {type_text}

Read the selected paragraphs and extract up to 12 atomic facts about the concept.

Rules:
- Each fact must be directly supported by the selected paragraphs.
- Each fact must be short and atomic: one property, part, action, function, habitat, sound, material, appearance, behavior, location, or mechanism.
- Prefer concrete facts that help distinguish this concept from many others.
- Avoid vague facts like "is common", "is useful", "is important", "has parts", or "can be found".
- Avoid pure taxonomy unless it is very informative.
- The fact must be handle-safe: it should be possible to write "{{handle}} ..." without naming the concept again.
- Do not include the concept name inside the fact string.
- Do not invent facts from outside the paragraphs.
- Quote the exact evidence sentence or phrase from the selected paragraphs.
- Return only valid JSON.

Return this shape:
{{
  "concept_id": "{concept_id}",
  "concept": "{concept}",
  "facts": [
    {{
      "fact": "...",
      "evidence": "...",
      "source_paragraph_id": "...",
      "fact_type": "part|material|action|function|habitat|sound|mechanism|appearance|behavior|location|other",
      "handle_safe": true,
      "specificity": "high|medium|low"
    }}
  ]
}}

Selected paragraphs:

{chr(10).join(paragraph_blocks)}"""
    if disable_thinking:
        prompt = f"{prompt}\n\n/no_think"
    return prompt


def build_template_prompt(
    *,
    concept_id: str,
    concept: str,
    facts: list[FactCandidate],
    disable_thinking: bool = True,
) -> str:
    fact_payload = [
        {
            "fact_id": fact.id,
            "fact": fact.fact,
            "fact_type": fact.fact_type,
            "evidence": fact.evidence,
        }
        for fact in facts
    ]
    prompt = f"""You are converting supported concept facts into handle-safe surface templates.

Concept ID: {concept_id}
Concept: {concept}

For each fact, write exactly 3 short templates:
- 2 direct templates using {{handle}} as the entity.
- 1 retrieval template like "The one that ... was {{handle}}."

Rules:
- Every template must include the exact string {{handle}}.
- Do not mention the concept word "{concept}".
- Preserve only the fact given; do not add new facts.
- Use natural simple English suitable for short story or dialogue data.
- Do not use "therefore".
- Do not use WUGS-style wording such as "A wug is a ..." or "A {{handle}} is a ...".
- Return only valid JSON.

Return this shape:
{{
  "concept_id": "{concept_id}",
  "concept": "{concept}",
  "items": [
    {{
      "fact_id": "...",
      "templates": [
        {{"template": "...", "template_type": "direct"}},
        {{"template": "...", "template_type": "direct"}},
        {{"template": "...", "template_type": "retrieval"}}
      ]
    }}
  ]
}}

Filtered facts JSON:
{json.dumps(fact_payload, ensure_ascii=False, indent=2)}"""
    if disable_thinking:
        prompt = f"{prompt}\n\n/no_think"
    return prompt


def parse_extracted_facts(
    *,
    raw: str,
    concept_id: str,
    concept: str,
    extraction_round: str,
) -> list[FactCandidate]:
    payload = parse_json_object(raw)
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("extraction response must include a facts list")
    facts = []
    for index, item in enumerate(raw_facts):
        if not isinstance(item, dict):
            continue
        facts.append(
            FactCandidate(
                id=f"{concept_id}__{extraction_round}__fact{index:03d}",
                concept_id=concept_id,
                concept=concept,
                extraction_round=extraction_round,
                fact=normalize_space(item.get("fact") or ""),
                evidence=normalize_space(item.get("evidence") or ""),
                source_paragraph_id=normalize_space(item.get("source_paragraph_id") or ""),
                fact_type=normalize_space(item.get("fact_type") or "other").lower(),
                handle_safe=bool(item.get("handle_safe")),
                specificity=normalize_space(item.get("specificity") or "").lower(),
                raw_index=index,
            )
        )
    return facts


def parse_templates(
    *,
    raw: str,
    concept_id: str,
    concept: str,
    facts_by_id: dict[str, FactCandidate],
) -> list[TemplateCandidate]:
    payload = parse_json_object(raw)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("template response must include an items list")
    templates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fact_id = normalize_space(item.get("fact_id") or "")
        fact = facts_by_id.get(fact_id)
        raw_templates = item.get("templates")
        if fact is None or not isinstance(raw_templates, list):
            continue
        for index, raw_template in enumerate(raw_templates):
            if isinstance(raw_template, dict):
                template = normalize_space(raw_template.get("template") or "")
                template_type = normalize_space(raw_template.get("template_type") or "direct").lower()
            else:
                template = normalize_space(raw_template)
                template_type = "direct"
            templates.append(
                TemplateCandidate(
                    id=f"{fact_id}__template{index:02d}",
                    concept_id=concept_id,
                    concept=concept,
                    fact_id=fact_id,
                    fact=fact.fact,
                    template=template,
                    template_type=template_type,
                    raw_index=index,
                )
            )
    return templates


def fact_rejection_reason(
    fact: FactCandidate,
    paragraphs: dict[str, SourceParagraph],
) -> str | None:
    if not fact.fact:
        return "missing_fact"
    if not fact.evidence:
        return "missing_evidence"
    if not fact.source_paragraph_id:
        return "missing_source_paragraph_id"
    if fact.source_paragraph_id not in paragraphs:
        return "unknown_source_paragraph_id"
    if not evidence_supported(fact.evidence, {fact.source_paragraph_id: paragraphs[fact.source_paragraph_id]}):
        return "unsupported_evidence"
    if not fact.handle_safe:
        return "not_handle_safe"
    if fact.specificity == "low":
        return "low_specificity"
    if concept_leaks(fact.fact, fact.concept):
        return "concept_leak"
    if TAXONOMY_ONLY_PATTERN.search(fact.fact):
        return "taxonomy_only"
    if any(pattern.search(fact.fact) for pattern in GENERIC_PATTERNS):
        return "generic_fact"
    return None


def filter_fact_candidates(
    facts: list[FactCandidate],
    paragraphs: list[SourceParagraph],
) -> tuple[list[FactCandidate], Counter[str]]:
    by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    accepted = []
    rejections: Counter[str] = Counter()
    for fact in facts:
        reason = fact_rejection_reason(fact, by_id)
        if reason is None:
            accepted.append(fact)
        else:
            rejections[reason] += 1
    return accepted, rejections


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_key(left).split())
    right_tokens = set(normalize_key(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def dedupe_and_rank_facts(
    facts: list[FactCandidate],
    *,
    max_facts: int = DEFAULT_MAX_FACTS,
) -> tuple[list[FactCandidate], Counter[str]]:
    deduped: list[FactCandidate] = []
    rejections: Counter[str] = Counter()
    for fact in facts:
        duplicate = False
        for kept in deduped:
            if normalize_key(fact.fact) == normalize_key(kept.fact):
                duplicate = True
                break
            if token_jaccard(fact.fact, kept.fact) >= 0.9:
                duplicate = True
                break
            if normalize_key(fact.evidence) == normalize_key(kept.evidence) and token_jaccard(
                fact.fact, kept.fact
            ) >= 0.65:
                duplicate = True
                break
        if duplicate:
            rejections["duplicate"] += 1
        else:
            deduped.append(fact)

    specificity_score = {"high": 2, "medium": 1}
    indexed = list(enumerate(deduped))
    selected: list[FactCandidate] = []
    used_ids: set[str] = set()
    for fact_type in sorted({fact.fact_type for fact in deduped}):
        typed = [(i, fact) for i, fact in indexed if fact.fact_type == fact_type and fact.id not in used_ids]
        if not typed:
            continue
        _, best = max(
            typed,
            key=lambda pair: (specificity_score.get(pair[1].specificity, 0), -pair[0]),
        )
        selected.append(best)
        used_ids.add(best.id)
        if len(selected) >= int(max_facts):
            return selected, rejections

    remaining = [pair for pair in indexed if pair[1].id not in used_ids]
    remaining.sort(key=lambda pair: (-specificity_score.get(pair[1].specificity, 0), pair[0]))
    for _, fact in remaining:
        if len(selected) >= int(max_facts):
            break
        selected.append(fact)
    if len(deduped) > len(selected):
        rejections["over_max_facts"] += len(deduped) - len(selected)
    return selected, rejections


def template_rejection_reason(template: TemplateCandidate) -> str | None:
    if not template.template:
        return "missing_template"
    if "{handle}" not in template.template:
        return "missing_handle"
    if concept_leaks(template.template, template.concept):
        return "concept_leak"
    if WUGS_TEMPLATE_PATTERN.search(template.template):
        return "wugs_like"
    return None


def filter_templates(
    templates: list[TemplateCandidate],
) -> tuple[list[TemplateCandidate], Counter[str]]:
    accepted: list[TemplateCandidate] = []
    rejections: Counter[str] = Counter()
    kept_per_fact: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for template in templates:
        reason = template_rejection_reason(template)
        key = (template.fact_id, normalize_key(template.template))
        if reason is None and key in seen:
            reason = "duplicate"
        if reason is None and kept_per_fact[template.fact_id] >= 3:
            reason = "over_max_templates_per_fact"
        if reason is None:
            accepted.append(template)
            seen.add(key)
            kept_per_fact[template.fact_id] += 1
        else:
            rejections[reason] += 1
    return accepted, rejections


class DeterministicPropertyBankRenderer:
    """Rule-based renderer for smoke tests and local dry-runs."""

    def __call__(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        stage: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del system_prompt, temperature, max_tokens, response_format
        if stage == "extraction":
            return json.dumps(self._extract(prompt), ensure_ascii=False)
        if stage == "templates":
            return json.dumps(self._templates(prompt), ensure_ascii=False)
        raise ValueError(f"Unsupported deterministic stage: {stage}")

    def _extract(self, prompt: str) -> dict[str, Any]:
        concept_id = _extract_prompt_field(prompt, "Concept ID")
        concept = _extract_prompt_field(prompt, "Concept")
        facts = []
        for paragraph_id, text in _extract_prompt_paragraphs(prompt):
            for sentence in sentence_split(text)[:4]:
                fact = _sentence_to_fact(sentence, concept)
                if not fact:
                    continue
                facts.append(
                    {
                        "fact": fact,
                        "evidence": sentence,
                        "source_paragraph_id": paragraph_id,
                        "fact_type": _guess_fact_type(fact),
                        "handle_safe": True,
                        "specificity": "high",
                    }
                )
                if len(facts) >= 12:
                    break
            if len(facts) >= 12:
                break
        return {"concept_id": concept_id, "concept": concept, "facts": facts}

    def _templates(self, prompt: str) -> dict[str, Any]:
        concept_id = _extract_prompt_field(prompt, "Concept ID")
        concept = _extract_prompt_field(prompt, "Concept")
        facts = _extract_fact_payload(prompt)
        items = []
        for fact in facts:
            fact_id = str(fact.get("fact_id") or "")
            fact_text = normalize_space(fact.get("fact") or "")
            predicate = _fact_to_predicate(fact_text)
            items.append(
                {
                    "fact_id": fact_id,
                    "templates": [
                        {"template": f"{{handle}} {predicate}.", "template_type": "direct"},
                        {"template": f"Everyone noticed that {{handle}} {predicate}.", "template_type": "direct"},
                        {"template": f"The one that {predicate} was {{handle}}.", "template_type": "retrieval"},
                    ],
                }
            )
        return {"concept_id": concept_id, "concept": concept, "items": items}


def _extract_prompt_field(prompt: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", prompt, flags=re.MULTILINE)
    return normalize_space(match.group(1)) if match else ""


def _extract_prompt_paragraphs(prompt: str) -> list[tuple[str, str]]:
    matches = list(
        re.finditer(
            r"Paragraph ID:\s*(?P<id>[^\n]+)\nStyle:\s*[^\n]*\nText:\s*(?P<text>.*?)(?=\nParagraph ID:|\n\n/no_think|\Z)",
            prompt,
            flags=re.DOTALL,
        )
    )
    return [(normalize_space(match.group("id")), normalize_space(match.group("text"))) for match in matches]


def _extract_fact_payload(prompt: str) -> list[dict[str, Any]]:
    marker = "Filtered facts JSON:"
    start = prompt.find(marker)
    if start < 0:
        return []
    text = prompt[start + len(marker) :].strip()
    if text.endswith("/no_think"):
        text = text[: -len("/no_think")].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _sentence_to_fact(sentence: str, concept: str) -> str:
    text = normalize_space(sentence).rstrip(".")
    for variant in concept_leak_variants(concept):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(variant)}(?![A-Za-z0-9])", "it", text, flags=re.I)
    text = re.sub(r"^(?:a|an|the)\s+it\s+", "it ", text, flags=re.I)
    if concept_leaks(text, concept):
        return ""
    return text[:180]


def _guess_fact_type(fact: str) -> str:
    lower = fact.lower()
    if any(word in lower for word in ("sound", "noise", "call", "tone", "song")):
        return "sound"
    if any(word in lower for word in ("moves", "walks", "flies", "swims", "runs", "grows")):
        return "action"
    if any(word in lower for word in ("uses", "used", "holds", "carries", "serves")):
        return "function"
    if any(word in lower for word in ("has", "features", "includes", "contains")):
        return "part"
    return "other"


def _fact_to_predicate(fact: str) -> str:
    text = normalize_space(fact).rstrip(".")
    text = re.sub(r"^(?:it|this one|the object|the animal)\s+", "", text, flags=re.I)
    if not text:
        return "had a supported property"
    if re.match(r"^(?:has|had|uses|used|moves|moved|makes|made|produces|produced|grows|grew|holds|held)\b", text, re.I):
        return text
    return f"had the property that {text}"


def create_renderer(config: PropertyBankConfig, *, api_key: str = "") -> PropertyBankRenderer | None:
    if config.renderer == PROMPT_ONLY:
        return None
    if config.renderer == DETERMINISTIC:
        return DeterministicPropertyBankRenderer()
    if config.renderer == FIREWORKS:
        return FireworksChatClient(
            model=config.model_id,
            api_key=api_key,
            timeout_seconds=config.timeout_seconds,
            max_request_retries=config.max_request_retries,
            reasoning_effort=config.reasoning_effort,
            reasoning_history=config.reasoning_history,
        )
    raise ValueError(f"renderer must be one of {sorted(RENDERER_TYPES)}")


def property_bank_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "source_paragraphs": output_dir / "source_paragraphs.jsonl",
        "extraction_prompts": output_dir / "extraction_prompts.jsonl",
        "extracted_facts_raw": output_dir / "extracted_facts_raw.jsonl",
        "extracted_facts_filtered": output_dir / "extracted_facts_filtered.jsonl",
        "surface_template_prompts": output_dir / "surface_template_prompts.jsonl",
        "surface_templates_raw": output_dir / "surface_templates_raw.jsonl",
        "surface_templates_filtered": output_dir / "surface_templates_filtered.jsonl",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
    }


def _select_concepts(config: PropertyBankConfig, concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if config.concept_ids:
        by_id = {str(row.get("concept_id")): row for row in concept_rows}
        missing = [concept_id for concept_id in config.concept_ids if concept_id not in by_id]
        if missing:
            raise ValueError(f"requested concept_id not found: {missing}")
        selected = [by_id[concept_id] for concept_id in config.concept_ids]
    else:
        selected = list(concept_rows)
    if config.concept_limit is not None:
        selected = selected[: max(0, int(config.concept_limit))]
    return selected


def build_property_bank(
    *,
    config: PropertyBankConfig,
    renderer: PropertyBankRenderer | None = None,
) -> PropertyBankResult:
    if config.renderer not in RENDERER_TYPES:
        raise ValueError(f"renderer must be one of {sorted(RENDERER_TYPES)}")
    concept_path = config.input_dir / config.by_concept_filename
    paragraph_path = config.input_dir / config.paragraph_filename
    concept_rows = read_jsonl(concept_path)
    paragraph_rows = read_jsonl(paragraph_path)
    paragraph_lookup = {str(row.get("id") or ""): row for row in paragraph_rows}
    selected_concepts = _select_concepts(config, concept_rows)
    rng = random.Random(int(config.seed))
    disable_thinking = str(config.reasoning_effort).lower() in {"none", "false", "0", "disabled"}

    renderer = renderer if renderer is not None else create_renderer(config)
    prompt_only = renderer is None

    source_paragraph_rows: list[dict[str, Any]] = []
    extraction_prompt_rows: list[dict[str, Any]] = []
    raw_fact_rows: list[dict[str, Any]] = []
    filtered_fact_rows: list[dict[str, Any]] = []
    template_prompt_rows: list[dict[str, Any]] = []
    raw_template_rows: list[dict[str, Any]] = []
    filtered_template_rows: list[dict[str, Any]] = []
    fact_rejections: Counter[str] = Counter()
    template_rejections: Counter[str] = Counter()
    parse_errors: list[dict[str, Any]] = []
    facts_by_concept: dict[str, list[FactCandidate]] = {}

    progress = tqdm(
        selected_concepts,
        desc="Property facts",
        unit="concept",
        dynamic_ncols=True,
        disable=prompt_only,
    )
    for concept_row in progress:
        concept_id = str(concept_row["concept_id"])
        concept = str(concept_row.get("concept") or concept_id.replace("_", " "))
        selected_paragraphs = select_initial_paragraphs(
            concept_row=concept_row,
            paragraph_lookup=paragraph_lookup,
            rng=rng,
        )
        all_selected_paragraphs = list(selected_paragraphs)
        for paragraph in selected_paragraphs:
            source_paragraph_rows.append(source_paragraph_to_row(paragraph))
        extraction_rounds = [("initial", selected_paragraphs)]
        accepted_candidates: list[FactCandidate] = []

        for round_name, paragraphs in extraction_rounds:
            prompt = build_extraction_prompt(
                concept_id=concept_id,
                concept=concept,
                concept_types=concept_types_from_row(concept_row),
                paragraphs=paragraphs,
                disable_thinking=disable_thinking,
            )
            prompt_record = PromptRecord(
                id=f"{concept_id}__{round_name}__extraction",
                concept_id=concept_id,
                concept=concept,
                stage="extraction",
                selection_round=round_name,
                paragraph_ids=tuple(paragraph.paragraph_id for paragraph in paragraphs),
                prompt=prompt,
            )
            extraction_prompt_rows.append(prompt_record_to_row(prompt_record))
            if prompt_only:
                continue
            assert renderer is not None
            try:
                raw = renderer(
                    prompt=prompt,
                    system_prompt="You extract supported atomic facts and return only JSON.",
                    temperature=float(config.extraction_temperature),
                    max_tokens=int(config.extraction_max_tokens),
                    stage="extraction",
                    response_format={"type": "json_object"},
                )
                facts = parse_extracted_facts(
                    raw=raw,
                    concept_id=concept_id,
                    concept=concept,
                    extraction_round=round_name,
                )
            except Exception as exc:  # noqa: BLE001 - recorded in audit for retry/review.
                parse_errors.append(
                    {
                        "concept_id": concept_id,
                        "concept": concept,
                        "stage": "extraction",
                        "selection_round": round_name,
                        "error": str(exc),
                    }
                )
                facts = []
            raw_fact_rows.extend(fact_to_row(fact) for fact in facts)
            accepted, rejections = filter_fact_candidates(facts, paragraphs)
            fact_rejections.update(rejections)
            accepted_candidates.extend(accepted)

        ranked, _ = dedupe_and_rank_facts(
            accepted_candidates,
            max_facts=int(config.max_facts_per_concept),
        )

        if (
            not prompt_only
            and len(ranked) < int(config.min_facts_per_concept)
            and len(all_selected_paragraphs) < int(concept_row.get("paragraph_count", 0) or 0)
        ):
            fill_in = select_fill_in_paragraphs(
                concept_row=concept_row,
                paragraph_lookup=paragraph_lookup,
                already_selected_ids={paragraph.paragraph_id for paragraph in all_selected_paragraphs},
                rng=rng,
                count=2,
            )
            for paragraph in fill_in:
                source_paragraph_rows.append(source_paragraph_to_row(paragraph))
            all_selected_paragraphs.extend(fill_in)
            if fill_in:
                round_name = "fill_in"
                prompt = build_extraction_prompt(
                    concept_id=concept_id,
                    concept=concept,
                    concept_types=concept_types_from_row(concept_row),
                    paragraphs=fill_in,
                    disable_thinking=disable_thinking,
                )
                prompt_record = PromptRecord(
                    id=f"{concept_id}__{round_name}__extraction",
                    concept_id=concept_id,
                    concept=concept,
                    stage="extraction",
                    selection_round=round_name,
                    paragraph_ids=tuple(paragraph.paragraph_id for paragraph in fill_in),
                    prompt=prompt,
                )
                extraction_prompt_rows.append(prompt_record_to_row(prompt_record))
                try:
                    assert renderer is not None
                    raw = renderer(
                        prompt=prompt,
                        system_prompt="You extract supported atomic facts and return only JSON.",
                        temperature=float(config.extraction_temperature),
                        max_tokens=int(config.extraction_max_tokens),
                        stage="extraction",
                        response_format={"type": "json_object"},
                    )
                    facts = parse_extracted_facts(
                        raw=raw,
                        concept_id=concept_id,
                        concept=concept,
                        extraction_round=round_name,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded in audit for retry/review.
                    parse_errors.append(
                        {
                            "concept_id": concept_id,
                            "concept": concept,
                            "stage": "extraction",
                            "selection_round": round_name,
                            "error": str(exc),
                        }
                    )
                    facts = []
                raw_fact_rows.extend(fact_to_row(fact) for fact in facts)
                accepted, rejections = filter_fact_candidates(facts, fill_in)
                fact_rejections.update(rejections)
                accepted_candidates.extend(accepted)

        ranked, rank_rejections = dedupe_and_rank_facts(
            accepted_candidates,
            max_facts=int(config.max_facts_per_concept),
        )
        fact_rejections.update(rank_rejections)
        facts_by_concept[concept_id] = ranked
        filtered_fact_rows.extend(fact_to_row(fact) for fact in ranked)

    if not prompt_only:
        progress_templates = tqdm(
            facts_by_concept.items(),
            desc="Property templates",
            unit="concept",
            dynamic_ncols=True,
        )
        for concept_id, facts in progress_templates:
            if not facts:
                continue
            concept = facts[0].concept
            prompt = build_template_prompt(
                concept_id=concept_id,
                concept=concept,
                facts=facts,
                disable_thinking=disable_thinking,
            )
            prompt_record = PromptRecord(
                id=f"{concept_id}__templates",
                concept_id=concept_id,
                concept=concept,
                stage="templates",
                selection_round="filtered_facts",
                paragraph_ids=(),
                prompt=prompt,
            )
            template_prompt_rows.append(prompt_record_to_row(prompt_record))
            try:
                assert renderer is not None
                raw = renderer(
                    prompt=prompt,
                    system_prompt="You convert supported facts into handle-safe templates and return only JSON.",
                    temperature=float(config.template_temperature),
                    max_tokens=int(config.template_max_tokens),
                    stage="templates",
                    response_format={"type": "json_object"},
                )
                parsed = parse_templates(
                    raw=raw,
                    concept_id=concept_id,
                    concept=concept,
                    facts_by_id={fact.id: fact for fact in facts},
                )
            except Exception as exc:  # noqa: BLE001 - recorded in audit for retry/review.
                parse_errors.append(
                    {
                        "concept_id": concept_id,
                        "concept": concept,
                        "stage": "templates",
                        "selection_round": "filtered_facts",
                        "error": str(exc),
                    }
                )
                parsed = []
            raw_template_rows.extend(template_to_row(template) for template in parsed)
            accepted_templates, rejections = filter_templates(parsed)
            template_rejections.update(rejections)
            filtered_template_rows.extend(template_to_row(template) for template in accepted_templates)

    outputs = property_bank_output_paths(config.output_dir)
    write_jsonl(outputs["source_paragraphs"], source_paragraph_rows)
    write_jsonl(outputs["extraction_prompts"], extraction_prompt_rows)
    write_jsonl(outputs["extracted_facts_raw"], raw_fact_rows)
    write_jsonl(outputs["extracted_facts_filtered"], filtered_fact_rows)
    write_jsonl(outputs["surface_template_prompts"], template_prompt_rows)
    write_jsonl(outputs["surface_templates_raw"], raw_template_rows)
    write_jsonl(outputs["surface_templates_filtered"], filtered_template_rows)

    fact_counts = Counter(row["concept_id"] for row in filtered_fact_rows)
    deficient = [
        {
            "concept_id": concept_id,
            "concept": next(
                (str(row.get("concept") or concept_id) for row in selected_concepts if row.get("concept_id") == concept_id),
                concept_id,
            ),
            "filtered_fact_count": fact_counts.get(concept_id, 0),
        }
        for concept_id in [str(row["concept_id"]) for row in selected_concepts]
        if fact_counts.get(concept_id, 0) < int(config.min_facts_per_concept)
    ]
    fact_count_values = [fact_counts.get(str(row["concept_id"]), 0) for row in selected_concepts]
    coverage = {
        "concepts_with_min_facts": len(selected_concepts) - len(deficient),
        "concepts_with_min_facts_fraction": (
            (len(selected_concepts) - len(deficient)) / len(selected_concepts) if selected_concepts else 0.0
        ),
        "deficient_concept_count": len(deficient),
        "min_filtered_facts_per_concept": min(fact_count_values) if fact_count_values else 0,
        "mean_filtered_facts_per_concept": (
            statistics.mean(fact_count_values) if fact_count_values else 0.0
        ),
    }
    counts = {
        "concepts": len(selected_concepts),
        "source_paragraphs": len(source_paragraph_rows),
        "extraction_prompts": len(extraction_prompt_rows),
        "raw_facts": len(raw_fact_rows),
        "filtered_facts": len(filtered_fact_rows),
        "surface_template_prompts": len(template_prompt_rows),
        "raw_templates": len(raw_template_rows),
        "filtered_templates": len(filtered_template_rows),
        "parse_errors": len(parse_errors),
    }
    audit_report = {
        "coverage": coverage,
        "counts": counts,
        "deficient_concepts": deficient,
        "fact_rejection_counts": dict(sorted(fact_rejections.items())),
        "parse_errors": parse_errors,
        "template_rejection_counts": dict(sorted(template_rejections.items())),
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "comps_property_bank_v1",
        "version": "v1_seed42" if int(config.seed) == DEFAULT_SEED else f"v1_seed{config.seed}",
        "seed": int(config.seed),
        "renderer": config.renderer,
        "model_id": config.model_id if config.renderer == FIREWORKS else None,
        "reasoning_effort": config.reasoning_effort,
        "reasoning_history": config.reasoning_history,
        "source_files": {
            "input_dir": str(config.input_dir),
            "by_concept": str(concept_path),
            "paragraphs": str(paragraph_path),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "sampling": {
            "initial": "parts_and_properties + specialized/mechanism + seeded random remaining",
            "fill_in": "up to 2 additional paragraphs for concepts below minimum retained facts",
            "min_facts_per_concept": int(config.min_facts_per_concept),
            "max_facts_per_concept": int(config.max_facts_per_concept),
        },
        "temperatures": {
            "extraction": float(config.extraction_temperature),
            "templates": float(config.template_temperature),
        },
        "counts": counts,
        "coverage": coverage,
    }
    write_json(outputs["audit_report"], audit_report)
    write_json(outputs["manifest"], manifest)
    return PropertyBankResult(outputs=outputs, manifest=manifest, audit_report=audit_report)


def default_output_dir(input_dir: Path, seed: int) -> Path:
    return input_dir / "property_bank" / f"v1_seed{int(seed)}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument("--by-concept-filename", default=DEFAULT_BY_CONCEPT)
    parser.add_argument("--paragraph-filename", default=DEFAULT_PARAGRAPHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--renderer", choices=sorted(RENDERER_TYPES), default=PROMPT_ONLY)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-file", default="fireworks_api.txt")
    parser.add_argument("--concept-limit", type=int)
    parser.add_argument("--concept-id", action="append", dest="concept_ids", default=[])
    parser.add_argument("--min-facts-per-concept", type=int, default=DEFAULT_MIN_FACTS)
    parser.add_argument("--max-facts-per-concept", type=int, default=DEFAULT_MAX_FACTS)
    parser.add_argument("--extraction-temperature", type=float, default=DEFAULT_EXTRACTION_TEMPERATURE)
    parser.add_argument("--template-temperature", type=float, default=DEFAULT_TEMPLATE_TEMPERATURE)
    parser.add_argument("--extraction-max-tokens", type=int, default=DEFAULT_EXTRACTION_MAX_TOKENS)
    parser.add_argument("--template-max-tokens", type=int, default=DEFAULT_TEMPLATE_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--reasoning-history", default="disabled")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-request-retries", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> tuple[PropertyBankConfig, str]:
    input_dir = resolve_path(args.input_dir, base_dir=ROOT)
    output_dir = (
        resolve_path(args.output_dir, base_dir=ROOT)
        if args.output_dir
        else default_output_dir(input_dir, args.seed)
    )
    api_key = ""
    if args.renderer == FIREWORKS:
        api_key_file = (
            str(resolve_path(args.api_key_file, base_dir=ROOT))
            if args.api_key_file
            else None
        )
        api_key = load_api_key(api_key=args.api_key, api_key_file=api_key_file)
    config = PropertyBankConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        by_concept_filename=str(args.by_concept_filename),
        paragraph_filename=str(args.paragraph_filename),
        seed=int(args.seed),
        renderer=str(args.renderer),
        model_id=str(args.model_id),
        concept_limit=args.concept_limit,
        concept_ids=tuple(str(item) for item in args.concept_ids),
        min_facts_per_concept=int(args.min_facts_per_concept),
        max_facts_per_concept=int(args.max_facts_per_concept),
        extraction_temperature=float(args.extraction_temperature),
        template_temperature=float(args.template_temperature),
        extraction_max_tokens=int(args.extraction_max_tokens),
        template_max_tokens=int(args.template_max_tokens),
        reasoning_effort=str(args.reasoning_effort),
        reasoning_history=str(args.reasoning_history),
        timeout_seconds=int(args.timeout_seconds),
        max_request_retries=int(args.max_request_retries),
        force=bool(args.force),
    )
    return config, api_key


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, api_key = config_from_args(args)
    renderer = create_renderer(config, api_key=api_key)
    result = build_property_bank(config=config, renderer=renderer)
    print(
        "COMPS property bank: "
        f"renderer={config.renderer} "
        f"concepts={result.manifest['counts']['concepts']} "
        f"facts={result.manifest['counts']['filtered_facts']} "
        f"templates={result.manifest['counts']['filtered_templates']} "
        f"output_dir={config.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
