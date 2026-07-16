"""Streaming corpus utilities for training BabyLM Word2Vec models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

from babysteps.data.sources import iter_dataset_rows


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TokenizationConfig:
    """Small, deterministic tokenizer settings for static-vector analysis."""

    lowercase: bool = True
    split_sentences: bool = True
    min_token_length: int = 1


def tokenization_config_from_mapping(config: dict[str, Any] | None) -> TokenizationConfig:
    """Build tokenizer settings from a YAML-compatible mapping."""

    config = config or {}
    return TokenizationConfig(
        lowercase=bool(config.get("lowercase", True)),
        split_sentences=bool(config.get("split_sentences", True)),
        min_token_length=int(config.get("min_token_length", 1)),
    )


def tokenize_text(text: str, config: TokenizationConfig) -> list[str]:
    """Tokenize one span into Word2Vec-style word tokens."""

    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if config.lowercase:
            token = token.lower()
        if len(token) < config.min_token_length:
            continue
        tokens.append(token)
    return tokens


def iter_sentence_texts(text: str, config: TokenizationConfig) -> Iterator[str]:
    """Yield sentence-like spans from a document or row text."""

    if not config.split_sentences:
        if text.strip():
            yield text
        return

    for sentence in SENTENCE_BOUNDARY_RE.split(text):
        sentence = sentence.strip()
        if sentence:
            yield sentence


class DatasetSentenceCorpus:
    """Restartable sentence iterator over a configured dataset.

    Gensim's training loop needs to iterate over the corpus more than once. This
    class deliberately reopens/restreams the dataset on every ``__iter__`` call.
    """

    def __init__(
        self,
        dataset_config: dict[str, Any],
        *,
        text_field: str = "text",
        tokenization: TokenizationConfig | None = None,
        max_docs: int | None = None,
    ) -> None:
        self.dataset_config = dict(dataset_config)
        self.text_field = text_field
        self.tokenization = tokenization or TokenizationConfig()
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[list[str]]:
        docs_seen = 0
        for row in iter_dataset_rows(self.dataset_config):
            if self.max_docs is not None and docs_seen >= self.max_docs:
                break
            if self.text_field not in row:
                raise KeyError(
                    f"Dataset row is missing text field {self.text_field!r}: {list(row.keys())}"
                )
            text = row[self.text_field]
            if text is None:
                continue
            docs_seen += 1
            for sentence in iter_sentence_texts(str(text), self.tokenization):
                tokens = tokenize_text(sentence, self.tokenization)
                if tokens:
                    yield tokens


def corpus_from_config(
    config: dict[str, Any],
    *,
    max_docs: int | None = None,
) -> DatasetSentenceCorpus:
    """Create a restartable corpus iterator from a Word2Vec training config."""

    dataset_config = dict(config["dataset"])
    preprocessing_config = config.get("preprocessing", {})
    text_field = str(dataset_config.get("text_field", "text"))
    return DatasetSentenceCorpus(
        dataset_config,
        text_field=text_field,
        tokenization=tokenization_config_from_mapping(preprocessing_config),
        max_docs=max_docs,
    )


__all__ = [
    "DatasetSentenceCorpus",
    "TokenizationConfig",
    "corpus_from_config",
    "iter_sentence_texts",
    "tokenization_config_from_mapping",
    "tokenize_text",
]
