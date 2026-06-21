"""Tokenizer fertility helpers for lexical analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenizationRecord:
    """Cached tokenizer pieces for one normalized word type."""

    word: str
    pieces: tuple[str, ...]

    @property
    def fertility(self) -> int:
        return len(self.pieces)

    @property
    def pieces_text(self) -> str:
        return " ".join(self.pieces)


class TokenizerAnalyzer:
    """Cache subword pieces and fertility for normalized word types."""

    def __init__(self, tokenizer: Any, *, mode: str = "leading_space") -> None:
        mode = mode.strip().lower()
        if mode not in {"leading_space", "isolated"}:
            raise ValueError("tokenizer_mode must be 'leading_space' or 'isolated'.")
        self.tokenizer = tokenizer
        self.mode = mode
        self._cache: dict[str, TokenizationRecord] = {}

    def analyze(self, word: str) -> TokenizationRecord:
        if word not in self._cache:
            text = f" {word}" if self.mode == "leading_space" else word
            pieces = tuple(str(piece) for piece in self._tokenize(text))
            self._cache[word] = TokenizationRecord(word=word, pieces=pieces)
        return self._cache[word]

    def _tokenize(self, text: str) -> list[str]:
        if hasattr(self.tokenizer, "tokenize"):
            try:
                return list(self.tokenizer.tokenize(text, add_special_tokens=False))
            except TypeError:
                return list(self.tokenizer.tokenize(text))

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return list(self.tokenizer.convert_ids_to_tokens(input_ids))
        return [str(token_id) for token_id in input_ids]


__all__ = ["TokenizationRecord", "TokenizerAnalyzer"]
