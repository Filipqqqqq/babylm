"""Reusable raw-text corpus mixing utilities.

The mixer keeps train.txt plain and moves provenance into sidecar metadata.
It is intentionally split into small components so sampling, ordering, word
counting, and writing can be tested or replaced independently.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from babysteps.paths import expand_env_vars, resolve_path


DEFAULT_WORD_PATTERN = r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*"


class WordCounter(Protocol):
    pattern: str

    def count(self, text: str) -> int:
        """Return the number of word-like units in ``text``."""

    def trim_to_words(self, text: str, max_words: int) -> str:
        """Return a prefix of ``text`` with at most ``max_words`` words."""


class DocumentReader(Protocol):
    def read_documents(self, source: "TextMixSourceConfig") -> list["SourceDocument"]:
        """Read source documents using the source config."""


class SourceSampler(Protocol):
    def sample(
        self,
        documents: list["SourceDocument"],
        *,
        source: "TextMixSourceConfig",
        seed: int,
    ) -> "SampledSource":
        """Select documents for one source."""


class DocumentOrderer(Protocol):
    def order(
        self,
        documents: list["SelectedDocument"],
        *,
        seed: int,
    ) -> list["SelectedDocument"]:
        """Return documents in final corpus order."""


@dataclass(frozen=True)
class TextMixSourceConfig:
    name: str
    role: str
    path: Path
    target_words: int
    sampling: str
    path_text: str


@dataclass(frozen=True)
class TextMixConfig:
    name: str
    output_dir: Path
    max_words: int
    seed: int
    strict_max_words: bool
    document_mode: str
    final_order: str
    trim_policy: str
    word_pattern: str
    sources: tuple[TextMixSourceConfig, ...]


@dataclass(frozen=True)
class SourceDocument:
    source_name: str
    source_role: str
    source_path: str
    source_doc_index: int
    line_number: int
    text: str
    word_count: int


@dataclass(frozen=True)
class SelectedDocument:
    document: SourceDocument
    repeat_index: int
    selection_index: int

    @property
    def word_count(self) -> int:
        return self.document.word_count


@dataclass(frozen=True)
class SampledSource:
    source: TextMixSourceConfig
    documents: tuple[SelectedDocument, ...]
    available_documents: int
    available_words: int
    selected_words: int
    full_repeats: int
    partial_repeat_documents: int


@dataclass(frozen=True)
class WrittenDocument:
    output_index: int
    selected_document: SelectedDocument
    output_words: int
    trimmed: bool
    text: str


@dataclass(frozen=True)
class WrittenCorpus:
    documents: tuple[WrittenDocument, ...]
    total_words: int
    trimmed_document: WrittenDocument | None
    skipped_selected_documents: int


@dataclass(frozen=True)
class TextMixResult:
    config: TextMixConfig
    sampled_sources: tuple[SampledSource, ...]
    selected_documents: tuple[SelectedDocument, ...]
    ordered_documents: tuple[SelectedDocument, ...]
    written_corpus: WrittenCorpus
    outputs: dict[str, Path]
    counts: dict[str, Any]
    manifest: dict[str, Any]


class RegexWordCounter:
    """Word counter and trimmer backed by one explicit regex."""

    def __init__(self, pattern: str = DEFAULT_WORD_PATTERN) -> None:
        self.pattern = pattern
        self._regex = re.compile(pattern)

    def count(self, text: str) -> int:
        return len(self._regex.findall(text))

    def trim_to_words(self, text: str, max_words: int) -> str:
        if max_words <= 0:
            return ""
        matches = list(self._regex.finditer(text))
        if len(matches) <= max_words:
            return text.strip()
        boundary = matches[max_words].start()
        return text[:boundary].strip()


class LineDocumentReader:
    """Read each non-empty text line as one document."""

    def __init__(self, counter: WordCounter, *, encoding: str = "utf-8") -> None:
        self.counter = counter
        self.encoding = encoding

    def read_documents(self, source: TextMixSourceConfig) -> list[SourceDocument]:
        if not source.path.is_file():
            raise FileNotFoundError(f"Missing text mix source: {source.path}")

        documents: list[SourceDocument] = []
        with source.path.open("r", encoding=self.encoding) as f:
            for line_number, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                word_count = self.counter.count(text)
                if word_count <= 0:
                    continue
                documents.append(
                    SourceDocument(
                        source_name=source.name,
                        source_role=source.role,
                        source_path=source.path_text,
                        source_doc_index=len(documents),
                        line_number=line_number,
                        text=text,
                        word_count=word_count,
                    )
                )

        if not documents:
            raise ValueError(f"Text mix source has no non-empty word documents: {source.path}")
        return documents


class RandomWithoutReplacementSampler:
    """Select whole documents from one seeded pseudo-random order."""

    def sample(
        self,
        documents: list[SourceDocument],
        *,
        source: TextMixSourceConfig,
        seed: int,
    ) -> SampledSource:
        ordered = list(documents)
        random.Random(seed).shuffle(ordered)
        selected: list[SelectedDocument] = []
        selected_words = 0
        for document in ordered:
            selected.append(
                SelectedDocument(
                    document=document,
                    repeat_index=0,
                    selection_index=len(selected),
                )
            )
            selected_words += document.word_count
            if selected_words >= source.target_words:
                break

        if selected_words < source.target_words:
            raise ValueError(
                f"Source {source.name!r} has {selected_words:,} available words after "
                f"sampling, below target {source.target_words:,}."
            )

        return SampledSource(
            source=source,
            documents=tuple(selected),
            available_documents=len(documents),
            available_words=sum(document.word_count for document in documents),
            selected_words=selected_words,
            full_repeats=0,
            partial_repeat_documents=0,
        )


class RepeatThenPartialSampler:
    """Repeat full corpora, then sample a seeded partial repeat."""

    def sample(
        self,
        documents: list[SourceDocument],
        *,
        source: TextMixSourceConfig,
        seed: int,
    ) -> SampledSource:
        available_words = sum(document.word_count for document in documents)
        if available_words <= 0:
            raise ValueError(f"Source {source.name!r} has zero words.")

        full_repeats = source.target_words // available_words
        selected: list[SelectedDocument] = []
        selected_words = 0

        for repeat_index in range(full_repeats):
            for document in documents:
                selected.append(
                    SelectedDocument(
                        document=document,
                        repeat_index=repeat_index,
                        selection_index=len(selected),
                    )
                )
                selected_words += document.word_count

        partial_docs = 0
        if selected_words < source.target_words:
            partial_order = list(documents)
            random.Random(seed).shuffle(partial_order)
            for document in partial_order:
                selected.append(
                    SelectedDocument(
                        document=document,
                        repeat_index=full_repeats,
                        selection_index=len(selected),
                    )
                )
                selected_words += document.word_count
                partial_docs += 1
                if selected_words >= source.target_words:
                    break

        if selected_words < source.target_words:
            raise ValueError(
                f"Source {source.name!r} could not reach target {source.target_words:,} "
                "with one partial repeat."
            )

        return SampledSource(
            source=source,
            documents=tuple(selected),
            available_documents=len(documents),
            available_words=available_words,
            selected_words=selected_words,
            full_repeats=full_repeats,
            partial_repeat_documents=partial_docs,
        )


class SeededDocumentShuffle:
    """Shuffle selected documents with a seeded pseudo-random order."""

    def order(
        self,
        documents: list[SelectedDocument],
        *,
        seed: int,
    ) -> list[SelectedDocument]:
        ordered = list(documents)
        random.Random(seed).shuffle(ordered)
        return ordered


class FinalCorpusWriter:
    """Write the capped train.txt and return document-level output records."""

    def __init__(self, counter: WordCounter, *, encoding: str = "utf-8") -> None:
        self.counter = counter
        self.encoding = encoding

    def write_train_text(
        self,
        ordered_documents: list[SelectedDocument],
        *,
        output_path: Path,
        max_words: int,
        strict_max_words: bool,
    ) -> WrittenCorpus:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written: list[WrittenDocument] = []
        total_words = 0
        trimmed_document: WrittenDocument | None = None

        with output_path.open("w", encoding=self.encoding) as f:
            for selected in ordered_documents:
                remaining = max_words - total_words
                if strict_max_words and remaining <= 0:
                    break

                text = selected.document.text
                output_words = selected.word_count
                trimmed = False
                if strict_max_words and output_words > remaining:
                    text = self.counter.trim_to_words(text, remaining)
                    output_words = self.counter.count(text)
                    trimmed = True
                    if output_words <= 0:
                        break

                record = WrittenDocument(
                    output_index=len(written),
                    selected_document=selected,
                    output_words=output_words,
                    trimmed=trimmed,
                    text=text,
                )
                f.write(text)
                f.write("\n")
                written.append(record)
                total_words += output_words

                if trimmed:
                    trimmed_document = record
                    break

        return WrittenCorpus(
            documents=tuple(written),
            total_words=total_words,
            trimmed_document=trimmed_document,
            skipped_selected_documents=len(ordered_documents) - len(written),
        )


class TextMixBuilder:
    """Coordinate source reading, source sampling, final ordering, and writing."""

    def __init__(
        self,
        *,
        reader: DocumentReader,
        samplers: dict[str, SourceSampler],
        orderer: DocumentOrderer,
        writer: FinalCorpusWriter,
    ) -> None:
        self.reader = reader
        self.samplers = samplers
        self.orderer = orderer
        self.writer = writer

    def build(
        self,
        config: TextMixConfig,
        *,
        config_path: Path | None = None,
    ) -> TextMixResult:
        output_dir = config.output_dir
        train_path = output_dir / "train.txt"
        config_out_path = output_dir / "mix_config.yaml"
        documents_path = output_dir / "mix_documents.jsonl"
        manifest_path = output_dir / "mix_manifest.json"
        counts_path = output_dir / "counts.json"

        sampled_sources: list[SampledSource] = []
        selected_documents: list[SelectedDocument] = []
        for source in config.sources:
            documents = self.reader.read_documents(source)
            sampler = self.samplers.get(source.sampling)
            if sampler is None:
                available = ", ".join(sorted(self.samplers))
                raise ValueError(
                    f"Unsupported sampling strategy {source.sampling!r}; available: {available}"
                )
            sampled = sampler.sample(
                documents,
                source=source,
                seed=config.seed,
            )
            sampled_sources.append(sampled)
            selected_documents.extend(sampled.documents)

        ordered_documents = self.orderer.order(
            selected_documents,
            seed=config.seed,
        )
        written_corpus = self.writer.write_train_text(
            ordered_documents,
            output_path=train_path,
            max_words=config.max_words,
            strict_max_words=config.strict_max_words,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            shutil.copyfile(config_path, config_out_path)

        counts = build_counts(config, sampled_sources, written_corpus)
        manifest = build_manifest(
            config=config,
            sampled_sources=sampled_sources,
            selected_documents=selected_documents,
            ordered_documents=ordered_documents,
            written_corpus=written_corpus,
            outputs={
                "train": train_path,
                "mix_config": config_out_path,
                "mix_manifest": manifest_path,
                "mix_documents": documents_path,
                "counts": counts_path,
            },
        )

        write_json(counts_path, counts)
        write_json(manifest_path, manifest)
        write_documents_jsonl(documents_path, written_corpus)

        return TextMixResult(
            config=config,
            sampled_sources=tuple(sampled_sources),
            selected_documents=tuple(selected_documents),
            ordered_documents=tuple(ordered_documents),
            written_corpus=written_corpus,
            outputs={
                "train": train_path,
                "mix_config": config_out_path,
                "mix_manifest": manifest_path,
                "mix_documents": documents_path,
                "counts": counts_path,
            },
            counts=counts,
            manifest=manifest,
        )


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def text_mix_config_from_mapping(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
) -> TextMixConfig:
    if not isinstance(mapping, dict):
        raise TypeError("Text mix config must be a mapping.")

    name = str(mapping["name"])
    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    max_words = int(mapping.get("max_words", 10_000_000))
    seed = int(mapping.get("seed", 42))
    strict_max_words = bool(mapping.get("strict_max_words", True))
    document_mode = str(mapping.get("document_mode", "line"))
    final_order = str(mapping.get("final_order", "deterministic_document_shuffle"))
    trim_policy = str(mapping.get("trim_policy", "trim_final_document_only_if_needed"))
    word_config = mapping.get("word_count") or {}
    word_pattern = str(word_config.get("pattern", DEFAULT_WORD_PATTERN))

    if document_mode != "line":
        raise ValueError("Text mixer currently supports only document_mode: line.")
    if final_order != "deterministic_document_shuffle":
        raise ValueError("Text mixer currently supports only deterministic_document_shuffle.")
    if trim_policy != "trim_final_document_only_if_needed":
        raise ValueError("Text mixer currently supports only trim_final_document_only_if_needed.")
    if max_words <= 0:
        raise ValueError("max_words must be positive.")

    sources = []
    raw_sources = mapping.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Text mix config requires a non-empty sources list.")
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise TypeError("Each source must be a mapping.")
        path_text = str(raw_source["path"])
        target_words = int(raw_source["target_words"])
        if target_words <= 0:
            raise ValueError("source.target_words must be positive.")
        sources.append(
            TextMixSourceConfig(
                name=str(raw_source["name"]),
                role=str(raw_source.get("role", "source")),
                path=resolve_path(path_text, base_dir=base_dir),
                target_words=target_words,
                sampling=str(raw_source["sampling"]),
                path_text=path_text,
            )
        )

    return TextMixConfig(
        name=name,
        output_dir=output_dir,
        max_words=max_words,
        seed=seed,
        strict_max_words=strict_max_words,
        document_mode=document_mode,
        final_order=final_order,
        trim_policy=trim_policy,
        word_pattern=word_pattern,
        sources=tuple(sources),
    )


def default_text_mix_builder(word_pattern: str = DEFAULT_WORD_PATTERN) -> TextMixBuilder:
    counter = RegexWordCounter(word_pattern)
    return TextMixBuilder(
        reader=LineDocumentReader(counter),
        samplers={
            "random_without_replacement": RandomWithoutReplacementSampler(),
            "repeat_then_partial": RepeatThenPartialSampler(),
        },
        orderer=SeededDocumentShuffle(),
        writer=FinalCorpusWriter(counter),
    )


def build_text_mix(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
) -> TextMixResult:
    config = text_mix_config_from_mapping(mapping, base_dir=base_dir)
    builder = default_text_mix_builder(config.word_pattern)
    return builder.build(config, config_path=config_path)


def build_counts(
    config: TextMixConfig,
    sampled_sources: list[SampledSource],
    written_corpus: WrittenCorpus,
) -> dict[str, Any]:
    output_by_source: dict[str, int] = {source.name: 0 for source in config.sources}
    output_docs_by_source: dict[str, int] = {source.name: 0 for source in config.sources}
    trimmed_source: str | None = None
    if written_corpus.trimmed_document is not None:
        trimmed_source = written_corpus.trimmed_document.selected_document.document.source_name

    for written in written_corpus.documents:
        name = written.selected_document.document.source_name
        output_by_source[name] += written.output_words
        output_docs_by_source[name] += 1

    source_counts = []
    for sampled in sampled_sources:
        source_counts.append(
            {
                "name": sampled.source.name,
                "role": sampled.source.role,
                "path": sampled.source.path_text,
                "target_words": sampled.source.target_words,
                "available_documents": sampled.available_documents,
                "available_words": sampled.available_words,
                "selected_documents": len(sampled.documents),
                "selected_words": sampled.selected_words,
                "output_documents": output_docs_by_source[sampled.source.name],
                "output_words": output_by_source[sampled.source.name],
                "output_fraction": (
                    output_by_source[sampled.source.name] / written_corpus.total_words
                    if written_corpus.total_words
                    else None
                ),
                "full_repeats": sampled.full_repeats,
                "partial_repeat_documents": sampled.partial_repeat_documents,
            }
        )

    selected_words_total = sum(sampled.selected_words for sampled in sampled_sources)
    return {
        "name": config.name,
        "seed": config.seed,
        "max_words": config.max_words,
        "strict_max_words": config.strict_max_words,
        "word_count_pattern": config.word_pattern,
        "target_words_total": sum(source.target_words for source in config.sources),
        "selected_words_total": selected_words_total,
        "output_words_total": written_corpus.total_words,
        "selected_documents_total": sum(len(sampled.documents) for sampled in sampled_sources),
        "output_documents_total": len(written_corpus.documents),
        "skipped_selected_documents": written_corpus.skipped_selected_documents,
        "trimmed": written_corpus.trimmed_document is not None,
        "trimmed_source": trimmed_source,
        "sources": source_counts,
    }


def build_manifest(
    *,
    config: TextMixConfig,
    sampled_sources: list[SampledSource],
    selected_documents: list[SelectedDocument],
    ordered_documents: list[SelectedDocument],
    written_corpus: WrittenCorpus,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.name,
        "seed": config.seed,
        "document_mode": config.document_mode,
        "final_order": config.final_order,
        "trim_policy": config.trim_policy,
        "strict_max_words": config.strict_max_words,
        "max_words": config.max_words,
        "word_count": {
            "kind": "regex",
            "pattern": config.word_pattern,
        },
        "sources": [
            {
                "name": sampled.source.name,
                "role": sampled.source.role,
                "path": sampled.source.path_text,
                "target_words": sampled.source.target_words,
                "sampling": sampled.source.sampling,
                "available_documents": sampled.available_documents,
                "available_words": sampled.available_words,
                "selected_documents": len(sampled.documents),
                "selected_words": sampled.selected_words,
                "full_repeats": sampled.full_repeats,
                "partial_repeat_documents": sampled.partial_repeat_documents,
            }
            for sampled in sampled_sources
        ],
        "totals": {
            "target_words": sum(source.target_words for source in config.sources),
            "selected_documents": len(selected_documents),
            "selected_words": sum(document.word_count for document in selected_documents),
            "ordered_documents": len(ordered_documents),
            "output_documents": len(written_corpus.documents),
            "output_words": written_corpus.total_words,
            "skipped_selected_documents": written_corpus.skipped_selected_documents,
        },
        "trimmed_document": (
            written_document_record(written_corpus.trimmed_document)
            if written_corpus.trimmed_document is not None
            else None
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def written_document_record(written: WrittenDocument | None) -> dict[str, Any] | None:
    if written is None:
        return None
    selected = written.selected_document
    document = selected.document
    return {
        "output_index": written.output_index,
        "source_name": document.source_name,
        "source_role": document.source_role,
        "source_path": document.source_path,
        "source_doc_index": document.source_doc_index,
        "line_number": document.line_number,
        "repeat_index": selected.repeat_index,
        "selection_index": selected.selection_index,
        "source_words": document.word_count,
        "output_words": written.output_words,
        "trimmed": written.trimmed,
    }


def write_documents_jsonl(path: Path, written_corpus: WrittenCorpus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for written in written_corpus.documents:
            payload = written_document_record(written)
            if payload is None:
                continue
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


__all__ = [
    "DEFAULT_WORD_PATTERN",
    "FinalCorpusWriter",
    "LineDocumentReader",
    "RandomWithoutReplacementSampler",
    "RegexWordCounter",
    "RepeatThenPartialSampler",
    "SeededDocumentShuffle",
    "SourceDocument",
    "TextMixBuilder",
    "TextMixConfig",
    "TextMixResult",
    "TextMixSourceConfig",
    "build_text_mix",
    "default_text_mix_builder",
    "load_yaml_config",
    "text_mix_config_from_mapping",
]
