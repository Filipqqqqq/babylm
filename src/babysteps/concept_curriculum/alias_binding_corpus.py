"""Generate balanced alias-binding corpus text from a COMPS property bank."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from babysteps.paths import resolve_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROPERTY_BANK_DIR = (
    "data/concept_curriculum/comps_concept_properties_clean_no_neighbors_full521_qwen/"
    "property_bank/v1_seed42"
)
DEFAULT_OUTPUT_DIR = "data/concept_curriculum/comps_alias_binding_v1_seed42"
DEFAULT_SEED = 42
DEFAULT_TARGET_WORDS = 100_000
DEFAULT_BINDINGS_PER_DOCUMENT = (2,)
SEQUENCE_FAMILIES = (
    "viewpoint_scene",
    "label_binding",
    "role_binding",
    "correction",
    "dialogue",
    "property_first",
)
CARRIER_TYPES = (
    "ordinary_name",
    "nonce_name",
    "color_nonce",
)
PROPERTY_ORDER_MODES = (
    "same_order",
    "reverse_order",
    "shuffled_order",
    "focus_first",
    "focus_last",
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
SPACE_PATTERN = re.compile(r"\s+")

ORDINARY_NAMES = (
    "Mira",
    "Nola",
    "Sari",
    "Tomas",
    "Kavi",
    "Pavo",
    "Rina",
    "Vera",
)
NONCE_NAMES = (
    "mip",
    "norb",
    "tavi",
    "lumo",
    "sile",
    "vimo",
    "paz",
    "kess",
)
COLOR_NONCE_NAMES = (
    "red mip",
    "blue norb",
    "green tavi",
    "yellow lumo",
    "white sile",
    "black vimo",
    "orange paz",
    "purple kess",
)
OBSERVERS = ("Jay", "Maya", "Lena", "Omar", "Nia", "Theo")


@dataclass(frozen=True)
class FactRecord:
    id: str
    concept_id: str
    concept: str
    fact: str
    fact_type: str
    evidence: str = ""


@dataclass(frozen=True)
class TemplateRecord:
    id: str
    concept_id: str
    concept: str
    fact_id: str
    fact: str
    fact_type: str
    template: str
    template_type: str


@dataclass(frozen=True)
class SamplingIndex:
    records: tuple[TemplateRecord, ...]
    by_concept: dict[str, tuple[TemplateRecord, ...]]
    concept_names: dict[str, str]
    concept_ids: tuple[str, ...]
    fact_types: tuple[str, ...]
    template_types: tuple[str, ...]


@dataclass(frozen=True)
class Binding:
    handle: str
    concept_id: str
    concept: str
    fact_id: str
    fact: str
    fact_type: str
    template_id: str
    template_type: str
    rendered_template: str


@dataclass(frozen=True)
class AliasBindingDocument:
    id: str
    text: str
    word_count: int
    sequence_family: str
    carrier_type: str
    property_order_mode: str
    property_order_handles: tuple[str, ...]
    bindings: tuple[Binding, ...]


@dataclass(frozen=True)
class AliasBindingConfig:
    property_bank_dir: Path
    output_dir: Path
    facts_filename: str = "extracted_facts_filtered.jsonl"
    templates_filename: str = "surface_templates_filtered.jsonl"
    seed: int = DEFAULT_SEED
    target_words: int = DEFAULT_TARGET_WORDS
    document_count: int | None = None
    bindings_per_document: tuple[int, ...] = DEFAULT_BINDINGS_PER_DOCUMENT
    sequence_families: tuple[str, ...] = SEQUENCE_FAMILIES
    carrier_types: tuple[str, ...] = CARRIER_TYPES
    property_order_modes: tuple[str, ...] = PROPERTY_ORDER_MODES


@dataclass(frozen=True)
class AliasBindingResult:
    documents: tuple[AliasBindingDocument, ...]
    manifest: dict[str, Any]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]


class SequenceRenderer(Protocol):
    family: str

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str: ...


def normalize_space(text: str) -> str:
    return SPACE_PATTERN.sub(" ", str(text).strip())


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def sentence_case(text: str) -> str:
    text = normalize_space(text)
    if not text:
        return text
    return f"{text[0].upper()}{text[1:]}"


def ensure_terminal_punctuation(text: str) -> str:
    text = normalize_space(text)
    if not text:
        return text
    return text if text[-1] in ".!?" else f"{text}."


def indefinite_article(text: str) -> str:
    first = normalize_space(text).lower()[:1]
    return "an" if first in {"a", "e", "i", "o", "u"} else "a"


def concept_with_article(concept: str) -> str:
    return f"{indefinite_article(concept)} {concept}"


def render_template(template: str, handle: str) -> str:
    rendered = normalize_space(template.replace("{handle}", handle))
    return ensure_terminal_punctuation(sentence_case(rendered))


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


def fact_record_from_row(row: dict[str, Any]) -> FactRecord:
    return FactRecord(
        id=str(row["id"]),
        concept_id=str(row["concept_id"]),
        concept=str(row.get("concept") or row["concept_id"]),
        fact=normalize_space(row.get("fact") or ""),
        fact_type=normalize_space(row.get("fact_type") or "other").lower(),
        evidence=normalize_space(row.get("evidence") or ""),
    )


def template_record_from_row(row: dict[str, Any], facts: dict[str, FactRecord]) -> TemplateRecord:
    fact_id = str(row["fact_id"])
    fact = facts.get(fact_id)
    fact_type = fact.fact_type if fact is not None else "other"
    evidence_fact = fact.fact if fact is not None else normalize_space(row.get("fact") or "")
    return TemplateRecord(
        id=str(row["id"]),
        concept_id=str(row["concept_id"]),
        concept=str(row.get("concept") or row["concept_id"]),
        fact_id=fact_id,
        fact=evidence_fact,
        fact_type=fact_type,
        template=normalize_space(row.get("template") or ""),
        template_type=normalize_space(row.get("template_type") or "direct").lower(),
    )


def load_sampling_index(
    *,
    facts_path: str | Path,
    templates_path: str | Path,
) -> SamplingIndex:
    facts = {fact.id: fact for fact in (fact_record_from_row(row) for row in read_jsonl(facts_path))}
    records = tuple(
        record
        for record in (template_record_from_row(row, facts) for row in read_jsonl(templates_path))
        if "{handle}" in record.template
    )
    if not records:
        raise ValueError(f"No usable templates found in {templates_path}")
    by_concept: dict[str, list[TemplateRecord]] = defaultdict(list)
    concept_names: dict[str, str] = {}
    for record in records:
        by_concept[record.concept_id].append(record)
        concept_names.setdefault(record.concept_id, record.concept)
    if len(by_concept) < 2:
        raise ValueError("Alias-binding sampling requires at least two concepts.")
    return SamplingIndex(
        records=records,
        by_concept={key: tuple(value) for key, value in sorted(by_concept.items())},
        concept_names=concept_names,
        concept_ids=tuple(sorted(by_concept)),
        fact_types=tuple(sorted({record.fact_type for record in records})),
        template_types=tuple(sorted({record.template_type for record in records})),
    )


def sampling_index_payload(index: SamplingIndex) -> dict[str, Any]:
    concepts: dict[str, Any] = {}
    for concept_id in index.concept_ids:
        facts: dict[str, Any] = {}
        for record in index.by_concept[concept_id]:
            fact_payload = facts.setdefault(
                record.fact_id,
                {
                    "fact": record.fact,
                    "fact_type": record.fact_type,
                    "templates": defaultdict(list),
                },
            )
            fact_payload["templates"][record.template_type].append(record.id)
        concepts[concept_id] = {
            "concept": index.concept_names[concept_id],
            "fact_count": len(facts),
            "template_count": len(index.by_concept[concept_id]),
            "facts": {
                fact_id: {
                    "fact": payload["fact"],
                    "fact_type": payload["fact_type"],
                    "templates": {key: tuple(value) for key, value in sorted(payload["templates"].items())},
                }
                for fact_id, payload in sorted(facts.items())
            },
        }
    return {
        "concept_count": len(index.concept_ids),
        "fact_types": index.fact_types,
        "template_types": index.template_types,
        "concepts": concepts,
    }


class RoundRobin:
    def __init__(self, values: tuple[Any, ...] | list[Any]) -> None:
        if not values:
            raise ValueError("RoundRobin requires at least one value.")
        self.values = tuple(values)
        self.index = 0

    def next(self) -> Any:
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return value


class BalancedAliasBindingSampler:
    def __init__(
        self,
        *,
        index: SamplingIndex,
        rng: random.Random,
        sequence_families: tuple[str, ...],
        carrier_types: tuple[str, ...],
        property_order_modes: tuple[str, ...],
    ) -> None:
        if not sequence_families:
            raise ValueError("At least one sequence family is required.")
        if not carrier_types:
            raise ValueError("At least one carrier type is required.")
        if not property_order_modes:
            raise ValueError("At least one property order mode is required.")
        unknown_order_modes = sorted(set(property_order_modes) - set(PROPERTY_ORDER_MODES))
        if unknown_order_modes:
            raise ValueError(f"Unknown property_order_modes: {unknown_order_modes}")
        self.index = index
        self.rng = rng
        self.concepts = RoundRobin(index.concept_ids)
        self.fact_types = RoundRobin(index.fact_types or ("other",))
        self.template_types = RoundRobin(index.template_types or ("direct",))
        self.sequence_families = RoundRobin(sequence_families)
        self.carrier_types = RoundRobin(carrier_types)
        self.property_order_modes = RoundRobin(property_order_modes)
        self.template_offsets: Counter[tuple[str, str, str]] = Counter()
        self.fallback_counts: Counter[str] = Counter()

    def sample_document(self, *, document_index: int, binding_count: int) -> AliasBindingDocument:
        if binding_count < 2:
            raise ValueError("binding_count must be at least 2.")
        if binding_count > len(self.index.concept_ids):
            raise ValueError("binding_count cannot exceed number of concepts.")
        sequence_family = str(self.sequence_families.next())
        carrier_type = str(self.carrier_types.next())
        property_order_mode = str(self.property_order_modes.next())
        concept_ids = self._sample_distinct_concepts(binding_count)
        handles = handles_for(carrier_type=carrier_type, count=binding_count, rng=self.rng)
        bindings = tuple(
            self._sample_binding(concept_id=concept_id, handle=handle)
            for concept_id, handle in zip(concept_ids, handles)
        )
        property_bindings = order_property_bindings(
            bindings=bindings,
            mode=property_order_mode,
            rng=self.rng,
        )
        renderer = renderer_for(sequence_family)
        text = renderer.render(
            bindings=bindings,
            property_bindings=property_bindings,
            carrier_type=carrier_type,
            rng=self.rng,
        )
        return AliasBindingDocument(
            id=f"alias_binding_{document_index:08d}",
            text=text,
            word_count=word_count(text),
            sequence_family=sequence_family,
            carrier_type=carrier_type,
            property_order_mode=property_order_mode,
            property_order_handles=tuple(binding.handle for binding in property_bindings),
            bindings=bindings,
        )

    def _sample_distinct_concepts(self, count: int) -> tuple[str, ...]:
        selected: list[str] = []
        while len(selected) < count:
            concept_id = str(self.concepts.next())
            if concept_id not in selected:
                selected.append(concept_id)
        return tuple(selected)

    def _sample_binding(self, *, concept_id: str, handle: str) -> Binding:
        desired_fact_type = str(self.fact_types.next())
        desired_template_type = str(self.template_types.next())
        records = self.index.by_concept[concept_id]
        candidates = [
            record
            for record in records
            if record.fact_type == desired_fact_type and record.template_type == desired_template_type
        ]
        fallback_reason = ""
        if not candidates:
            candidates = [record for record in records if record.template_type == desired_template_type]
            fallback_reason = "fact_type"
        if not candidates:
            candidates = [record for record in records if record.fact_type == desired_fact_type]
            fallback_reason = "template_type"
        if not candidates:
            candidates = list(records)
            fallback_reason = "fact_and_template_type"
        if fallback_reason:
            self.fallback_counts[fallback_reason] += 1
        key = (concept_id, desired_fact_type, desired_template_type)
        offset = self.template_offsets[key]
        record = candidates[offset % len(candidates)]
        self.template_offsets[key] += 1
        return Binding(
            handle=handle,
            concept_id=concept_id,
            concept=record.concept,
            fact_id=record.fact_id,
            fact=record.fact,
            fact_type=record.fact_type,
            template_id=record.id,
            template_type=record.template_type,
            rendered_template=render_template(record.template, handle),
        )


def handles_for(*, carrier_type: str, count: int, rng: random.Random) -> tuple[str, ...]:
    pools = {
        "ordinary_name": ORDINARY_NAMES,
        "nonce_name": NONCE_NAMES,
        "color_nonce": COLOR_NONCE_NAMES,
    }
    pool = pools.get(carrier_type)
    if pool is None:
        raise ValueError(f"Unknown carrier_type: {carrier_type}")
    if count > len(pool):
        raise ValueError(f"carrier_type {carrier_type!r} only has {len(pool)} handles.")
    return tuple(rng.sample(tuple(pool), k=count))


def order_property_bindings(
    *,
    bindings: tuple[Binding, ...],
    mode: str,
    rng: random.Random,
) -> tuple[Binding, ...]:
    if mode == "same_order":
        return bindings
    if mode == "reverse_order":
        return tuple(reversed(bindings))
    if mode == "shuffled_order":
        shuffled = list(bindings)
        rng.shuffle(shuffled)
        if len(shuffled) > 1 and tuple(shuffled) == bindings:
            shuffled = shuffled[1:] + shuffled[:1]
        return tuple(shuffled)
    if mode in {"focus_first", "focus_last"}:
        rest = list(bindings)
        focus = rest.pop(rng.randrange(len(rest)))
        rng.shuffle(rest)
        ordered = [focus, *rest] if mode == "focus_first" else [*rest, focus]
        return tuple(ordered)
    raise ValueError(f"Unknown property_order_mode: {mode}")


def binding_relation(*, carrier_type: str, rng: random.Random) -> str:
    if carrier_type == "ordinary_name":
        return rng.choice(("played as", "acted as"))
    return rng.choice(("stood for", "meant"))


def binding_list(bindings: tuple[Binding, ...], *, relation: str = "stood for") -> str:
    chunks = [
        f"{binding.handle} {relation} {concept_with_article(binding.concept)}"
        for binding in bindings
    ]
    if len(chunks) == 2:
        return f"{chunks[0]} and {chunks[1]}"
    return f"{', '.join(chunks[:-1])}, and {chunks[-1]}"


class ViewpointSceneRenderer:
    family = "viewpoint_scene"

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str:
        observer = rng.choice(OBSERVERS)
        relation = binding_relation(carrier_type=carrier_type, rng=rng)
        return normalize_space(
            f"{observer} watched a practice scene where {binding_list(bindings, relation=relation)}. "
            + " ".join(binding.rendered_template for binding in property_bindings)
        )


class LabelBindingRenderer:
    family = "label_binding"

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str:
        relation = binding_relation(carrier_type=carrier_type, rng=rng)
        return normalize_space(
            f"On the chart, {binding_list(bindings, relation=relation)}. "
            + " ".join(binding.rendered_template for binding in property_bindings)
        )


class RoleBindingRenderer:
    family = "role_binding"

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str:
        relation = binding_relation(carrier_type=carrier_type, rng=rng)
        return normalize_space(
            f"In the game, {binding_list(bindings, relation=relation)}. "
            + " ".join(binding.rendered_template for binding in property_bindings)
        )


class CorrectionRenderer:
    family = "correction"

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str:
        observer = rng.choice(OBSERVERS)
        first = property_bindings[0]
        relation = binding_relation(carrier_type=carrier_type, rng=rng)
        corrected = sentence_case(binding_list(bindings, relation=relation))
        return normalize_space(
            f"{observer} first mixed up the labels, then corrected the list. "
            f"{corrected}. "
            f"The corrected note focused on {first.handle}. "
            + " ".join(binding.rendered_template for binding in property_bindings)
        )


class DialogueRenderer:
    family = "dialogue"

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str:
        observer, questioner = rng.sample(tuple(OBSERVERS), k=2)
        relation = binding_relation(carrier_type=carrier_type, rng=rng)
        parts = [f"{observer} reviewed the mappings: {binding_list(bindings, relation=relation)}."]
        for binding in property_bindings:
            parts.append(f'"What about {binding.handle}?" {questioner} asked.')
            parts.append(f'"{binding.rendered_template}" {observer} said.')
        return normalize_space(" ".join(parts))


class PropertyFirstRenderer:
    family = "property_first"

    def render(
        self,
        *,
        bindings: tuple[Binding, ...],
        property_bindings: tuple[Binding, ...],
        carrier_type: str,
        rng: random.Random,
    ) -> str:
        relation = binding_relation(carrier_type=carrier_type, rng=rng)
        parts = [
            f"Two aliases were written on the board: {binding_list(bindings, relation=relation)}."
        ]
        for binding in property_bindings:
            parts.append("A clue came next.")
            parts.append(binding.rendered_template)
        return normalize_space(" ".join(parts))


def renderer_for(sequence_family: str) -> SequenceRenderer:
    renderers: dict[str, SequenceRenderer] = {
        ViewpointSceneRenderer.family: ViewpointSceneRenderer(),
        LabelBindingRenderer.family: LabelBindingRenderer(),
        RoleBindingRenderer.family: RoleBindingRenderer(),
        CorrectionRenderer.family: CorrectionRenderer(),
        DialogueRenderer.family: DialogueRenderer(),
        PropertyFirstRenderer.family: PropertyFirstRenderer(),
    }
    try:
        return renderers[sequence_family]
    except KeyError as exc:
        raise ValueError(f"Unknown sequence_family: {sequence_family}") from exc


def document_to_row(document: AliasBindingDocument) -> dict[str, Any]:
    row = asdict(document)
    row["bindings"] = [asdict(binding) for binding in document.bindings]
    row["source_template_ids"] = [binding.template_id for binding in document.bindings]
    row["concept_ids"] = [binding.concept_id for binding in document.bindings]
    row["fact_ids"] = [binding.fact_id for binding in document.bindings]
    return row


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "sampling_index": output_dir / "sampling_index.json",
        "manifest": output_dir / "manifest.json",
        "audit_report": output_dir / "audit_report.json",
    }


def build_alias_binding_corpus(config: AliasBindingConfig) -> AliasBindingResult:
    facts_path = config.property_bank_dir / config.facts_filename
    templates_path = config.property_bank_dir / config.templates_filename
    index = load_sampling_index(facts_path=facts_path, templates_path=templates_path)
    rng = random.Random(int(config.seed))
    sampler = BalancedAliasBindingSampler(
        index=index,
        rng=rng,
        sequence_families=config.sequence_families,
        carrier_types=config.carrier_types,
        property_order_modes=config.property_order_modes,
    )
    documents: list[AliasBindingDocument] = []
    total_words = 0
    binding_counts = RoundRobin(config.bindings_per_document)
    target_docs = config.document_count
    while True:
        if target_docs is not None and len(documents) >= target_docs:
            break
        if target_docs is None and total_words >= int(config.target_words):
            break
        document = sampler.sample_document(
            document_index=len(documents),
            binding_count=int(binding_counts.next()),
        )
        documents.append(document)
        total_words += document.word_count

    paths = output_paths(config.output_dir)
    write_jsonl(paths["documents"], [document_to_row(document) for document in documents])
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    paths["train"].write_text(
        "".join(f"{document.text}\n" for document in documents),
        encoding="utf-8",
    )
    write_json(paths["sampling_index"], sampling_index_payload(index))

    concept_counts: Counter[str] = Counter()
    template_type_counts: Counter[str] = Counter()
    fact_type_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    carrier_counts: Counter[str] = Counter()
    property_order_counts: Counter[str] = Counter()
    for document in documents:
        sequence_counts[document.sequence_family] += 1
        carrier_counts[document.carrier_type] += 1
        property_order_counts[document.property_order_mode] += 1
        for binding in document.bindings:
            concept_counts[binding.concept_id] += 1
            template_type_counts[binding.template_type] += 1
            fact_type_counts[binding.fact_type] += 1

    counts = {
        "documents": len(documents),
        "words": total_words,
        "bindings": sum(concept_counts.values()),
        "source_concepts": len(index.concept_ids),
        "source_templates": len(index.records),
    }
    balance = {
        "carrier_types": dict(sorted(carrier_counts.items())),
        "concepts": dict(sorted(concept_counts.items())),
        "fact_types": dict(sorted(fact_type_counts.items())),
        "property_order_modes": dict(sorted(property_order_counts.items())),
        "sequence_families": dict(sorted(sequence_counts.items())),
        "template_types": dict(sorted(template_type_counts.items())),
        "fallbacks": dict(sorted(sampler.fallback_counts.items())),
    }
    audit_report = {
        "balance": balance,
        "counts": counts,
        "target_words": int(config.target_words),
        "document_count_target": target_docs,
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "comps_alias_binding_corpus",
        "seed": int(config.seed),
        "source_files": {
            "property_bank_dir": str(config.property_bank_dir),
            "facts": str(facts_path),
            "templates": str(templates_path),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
        "counts": counts,
        "sampling": {
            "bindings_per_document": tuple(int(value) for value in config.bindings_per_document),
            "carrier_types": config.carrier_types,
            "property_order_modes": config.property_order_modes,
            "sequence_families": config.sequence_families,
            "strategy": (
                "round_robin_concepts_template_types_fact_types_carriers_sequence_families_"
                "and_property_order_modes"
            ),
        },
        "balance": balance,
    }
    write_json(paths["audit_report"], audit_report)
    write_json(paths["manifest"], manifest)
    return AliasBindingResult(
        documents=tuple(documents),
        manifest=manifest,
        audit_report=audit_report,
        outputs=paths,
    )


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("bindings per document must be >= 2")
    return values


def parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property-bank-dir", default=DEFAULT_PROPERTY_BANK_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--facts-filename", default="extracted_facts_filtered.jsonl")
    parser.add_argument("--templates-filename", default="surface_templates_filtered.jsonl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-words", type=int, default=DEFAULT_TARGET_WORDS)
    parser.add_argument("--document-count", type=int)
    parser.add_argument("--bindings-per-document", type=parse_int_tuple, default=DEFAULT_BINDINGS_PER_DOCUMENT)
    parser.add_argument("--sequence-families", type=parse_str_tuple, default=SEQUENCE_FAMILIES)
    parser.add_argument("--carrier-types", type=parse_str_tuple, default=CARRIER_TYPES)
    parser.add_argument("--property-order-modes", type=parse_str_tuple, default=PROPERTY_ORDER_MODES)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AliasBindingConfig:
    return AliasBindingConfig(
        property_bank_dir=resolve_path(args.property_bank_dir, base_dir=ROOT),
        output_dir=resolve_path(args.output_dir, base_dir=ROOT),
        facts_filename=str(args.facts_filename),
        templates_filename=str(args.templates_filename),
        seed=int(args.seed),
        target_words=int(args.target_words),
        document_count=args.document_count,
        bindings_per_document=tuple(args.bindings_per_document),
        sequence_families=tuple(args.sequence_families),
        carrier_types=tuple(args.carrier_types),
        property_order_modes=tuple(args.property_order_modes),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    result = build_alias_binding_corpus(config)
    print(
        "COMPS alias-binding corpus: "
        f"documents={result.manifest['counts']['documents']} "
        f"words={result.manifest['counts']['words']} "
        f"output_dir={config.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
