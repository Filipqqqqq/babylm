"""Text processing abstractions for lexical analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from babysteps.analysis.lexical.targets import normalize_lexeme


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class WordToken:
    """One normalized word token inside a sentence."""

    surface: str
    normalized: str
    lemma: str
    position: int


@dataclass(frozen=True)
class SentenceRecord:
    """Sentence text plus its normalized word tokens."""

    text: str
    tokens: tuple[WordToken, ...]


class TextProcessor(Protocol):
    """Protocol for text processors used by the lexical pipeline."""

    def process(self, text: str) -> list[SentenceRecord]:
        """Split text into sentences and normalized word tokens."""

    def pipe(self, texts: Iterable[str]) -> Iterator[list[SentenceRecord]]:
        """Process texts as a stream, yielding sentence records in order."""


def has_alpha(value: str) -> bool:
    """Return whether a normalized token contains at least one letter."""

    return any(ch.isalpha() for ch in value)


class RegexTextProcessor:
    """Small deterministic processor used for tests and no-lemma smoke runs."""

    def process(self, text: str) -> list[SentenceRecord]:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(text)
            if sentence.strip()
        ]
        if not sentences and text.strip():
            sentences = [text.strip()]

        records: list[SentenceRecord] = []
        for sentence in sentences:
            tokens = []
            for position, match in enumerate(_WORD_RE.finditer(sentence)):
                normalized = normalize_lexeme(match.group(0))
                if not normalized or not has_alpha(normalized):
                    continue
                tokens.append(
                    WordToken(
                        surface=match.group(0),
                        normalized=normalized,
                        lemma=normalized,
                        position=position,
                    )
                )
            records.append(SentenceRecord(text=sentence, tokens=tuple(tokens)))
        return records

    def pipe(self, texts: Iterable[str]) -> Iterator[list[SentenceRecord]]:
        for text in texts:
            yield self.process(text)


class SpacyTextProcessor:
    """spaCy-backed English tokenizer, sentence splitter, and lemmatizer."""

    def __init__(
        self,
        *,
        model: str = "en_core_web_sm",
        batch_size: int = 64,
        n_process: int = 4,
        disable: Iterable[str] = ("ner",),
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("spaCy batch_size must be positive.")
        if int(n_process) <= 0:
            raise ValueError("spaCy n_process must be positive.")

        self.batch_size = int(batch_size)
        self.n_process = int(n_process)
        self.disable = list(disable)

        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for lemmatized lexical analysis. "
                "Install the analysis extra, then download an English model: "
                "python -m spacy download en_core_web_sm"
            ) from exc

        try:
            self.nlp = spacy.load(model, disable=self.disable)
        except OSError as exc:
            raise OSError(
                f"Could not load spaCy model {model!r}. Install it with: "
                f"python -m spacy download {model}"
            ) from exc

        if not any(pipe in self.nlp.pipe_names for pipe in ("parser", "senter", "sentencizer")):
            self.nlp.add_pipe("sentencizer")

    def process(self, text: str) -> list[SentenceRecord]:
        doc = self.nlp(text)
        return self._doc_to_records(doc)

    def pipe(self, texts: Iterable[str]) -> Iterator[list[SentenceRecord]]:
        docs = self.nlp.pipe(
            texts,
            batch_size=self.batch_size,
            n_process=self.n_process,
        )
        for doc in docs:
            yield self._doc_to_records(doc)

    def _doc_to_records(self, doc) -> list[SentenceRecord]:
        spans = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
        records: list[SentenceRecord] = []
        for span in spans:
            tokens = []
            word_position = 0
            for token in span:
                normalized = normalize_lexeme(token.text)
                if not normalized or not has_alpha(normalized):
                    continue
                raw_lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
                lemma = normalize_lexeme(raw_lemma) or normalized
                tokens.append(
                    WordToken(
                        surface=token.text,
                        normalized=normalized,
                        lemma=lemma,
                        position=word_position,
                    )
                )
                word_position += 1
            sentence_text = span.text.strip()
            if sentence_text:
                records.append(SentenceRecord(text=sentence_text, tokens=tuple(tokens)))
        return records


def make_text_processor(config: dict[str, object] | None = None) -> TextProcessor:
    """Build the configured text processor."""

    config = config or {}
    kind = str(config.get("kind", "spacy")).strip().lower()
    if kind == "spacy":
        raw_disable = config.get("disable", ["ner"])
        if isinstance(raw_disable, str):
            disable = [item for item in raw_disable.replace(",", " ").split() if item]
        elif isinstance(raw_disable, Iterable):
            disable = [str(item) for item in raw_disable]
        else:
            raise TypeError("lemmatizer.disable must be a string or list of strings.")
        return SpacyTextProcessor(
            model=str(config.get("model", "en_core_web_sm")),
            batch_size=int(config.get("batch_size", 64)),
            n_process=int(config.get("n_process", 4)),
            disable=disable,
        )
    if kind in {"regex", "none"}:
        return RegexTextProcessor()
    raise ValueError(f"Unsupported lexical lemmatizer kind {kind!r}; expected spacy or regex.")


__all__ = [
    "RegexTextProcessor",
    "SentenceRecord",
    "SpacyTextProcessor",
    "TextProcessor",
    "WordToken",
    "has_alpha",
    "make_text_processor",
]
