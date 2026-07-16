"""Target-list parsing and matching for lexical analysis reports."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


_SPACE_RE = re.compile(r"\s+")
_EDGE_RE = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def normalize_lexeme(value: Any) -> str:
    """Normalize one English lexical item for counting and matching."""

    text = str(value).strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _SPACE_RE.sub(" ", text)
    text = _EDGE_RE.sub("", text)
    return text


def _validate_single_token(value: str, *, target_id: str, field: str) -> None:
    if not value:
        return
    if any(ch.isspace() for ch in value):
        raise ValueError(
            f"Target {target_id!r} has multiword {field} entry {value!r}; "
            "v1 lexical targets must be single-token entries."
        )


@dataclass(frozen=True)
class LexicalTarget:
    """One target concept represented by surface forms and optional lemmas."""

    id: str
    label: str
    surfaces: tuple[str, ...]
    lemmas: tuple[str, ...]
    target_type: Literal["word", "phrase"] = "word"


@dataclass(frozen=True)
class TargetMatch:
    """One target match over a contiguous sentence-token span."""

    target_id: str
    match_kinds: frozenset[str]
    start_index: int
    end_index: int
    surface: str
    lemma: str


def lexeme_tokens(value: str) -> tuple[str, ...]:
    normalized = normalize_lexeme(value)
    return tuple(part for part in normalized.split(" ") if part)


def _normalized_values(
    values: Any,
    *,
    target_id: str,
    field: str,
    allow_multiword: bool = False,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise TypeError(
            f"Target {target_id!r} field {field!r} must be a string or list of strings."
        )

    normalized = []
    for value in values:
        item = normalize_lexeme(value)
        if not allow_multiword:
            _validate_single_token(item, target_id=target_id, field=field)
        if item:
            normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def parse_targets(payload: dict[str, Any]) -> list[LexicalTarget]:
    """Parse and validate a lexical target YAML payload."""

    version = int(payload.get("version", 1))
    if version not in {1, 2}:
        raise ValueError(
            f"Unsupported lexical target version {version}; expected 1 or 2."
        )

    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise TypeError("Target YAML must contain a list field named 'targets'.")

    targets: list[LexicalTarget] = []
    seen_ids: set[str] = set()
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise TypeError(f"Target at index {index} must be a mapping.")

        target_id = normalize_lexeme(raw_target.get("id", ""))
        if not target_id:
            raise ValueError(f"Target at index {index} is missing a non-empty id.")
        _validate_single_token(target_id, target_id=target_id, field="id")
        if target_id in seen_ids:
            raise ValueError(f"Duplicate target id: {target_id!r}")
        seen_ids.add(target_id)

        raw_target_type = raw_target.get("target_type", "word")
        target_type = str(raw_target_type).strip().lower()
        if target_type not in {"word", "phrase"}:
            raise ValueError(
                f"Target {target_id!r} has unsupported target_type {raw_target_type!r}; "
                "expected 'word' or 'phrase'."
            )
        if version == 1 and target_type != "word":
            raise ValueError("v1 lexical targets only support target_type='word'.")

        label = str(raw_target.get("label", target_id)).strip() or target_id
        allow_multiword = target_type == "phrase"
        surfaces = _normalized_values(
            raw_target.get("surfaces", [label]),
            target_id=target_id,
            field="surfaces",
            allow_multiword=allow_multiword,
        )
        lemmas = _normalized_values(
            raw_target.get("lemmas", []),
            target_id=target_id,
            field="lemmas",
            allow_multiword=allow_multiword,
        )
        if not surfaces and not lemmas:
            raise ValueError(
                f"Target {target_id!r} must define at least one surface or lemma entry."
            )
        if target_type == "phrase":
            for field, values in (("surfaces", surfaces), ("lemmas", lemmas)):
                for value in values:
                    if not lexeme_tokens(value):
                        raise ValueError(
                            f"Target {target_id!r} has empty phrase {field} entry."
                        )

        targets.append(
            LexicalTarget(
                id=target_id,
                label=label,
                surfaces=surfaces,
                lemmas=lemmas,
                target_type=target_type,  # type: ignore[arg-type]
            )
        )

    return targets


def load_targets(path: str | Path) -> list[LexicalTarget]:
    """Load lexical targets from a YAML file."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Target YAML must contain a mapping: {path}")
    return parse_targets(payload)


class TargetMatcher:
    """Map normalized surface and lemma forms to lexical target ids."""

    def __init__(self, targets: list[LexicalTarget]) -> None:
        self.targets = {target.id: target for target in targets}
        self.surface_to_targets: dict[str, set[str]] = {}
        self.lemma_to_targets: dict[str, set[str]] = {}
        self.phrase_surface_to_targets: dict[tuple[str, ...], set[str]] = {}
        self.phrase_lemma_to_targets: dict[tuple[str, ...], set[str]] = {}
        self.max_phrase_length = 0
        for target in targets:
            if target.target_type == "phrase":
                for surface in target.surfaces:
                    parts = lexeme_tokens(surface)
                    if parts:
                        self.phrase_surface_to_targets.setdefault(parts, set()).add(
                            target.id
                        )
                        self.max_phrase_length = max(self.max_phrase_length, len(parts))
                for lemma in target.lemmas:
                    parts = lexeme_tokens(lemma)
                    if parts:
                        self.phrase_lemma_to_targets.setdefault(parts, set()).add(
                            target.id
                        )
                        self.max_phrase_length = max(self.max_phrase_length, len(parts))
                continue
            for surface in target.surfaces:
                self.surface_to_targets.setdefault(surface, set()).add(target.id)
            for lemma in target.lemmas:
                self.lemma_to_targets.setdefault(lemma, set()).add(target.id)

    def match(self, *, surface: str, lemma: str) -> dict[str, set[str]]:
        surface_hits = set(self.surface_to_targets.get(surface, set()))
        lemma_hits = set(self.lemma_to_targets.get(lemma, set()))
        target_ids = surface_hits | lemma_hits
        return {
            target_id: {
                kind
                for kind, hits in (
                    ("surface", surface_hits),
                    ("lemma", lemma_hits),
                )
                if target_id in hits
            }
            for target_id in target_ids
        }

    def match_sentence(self, tokens: Sequence[Any]) -> list[TargetMatch]:
        """Return word and phrase matches over one sentence token sequence."""

        matches: list[TargetMatch] = []
        surfaces = [str(token.normalized) for token in tokens]
        lemmas = [str(token.lemma) for token in tokens]

        for index, token in enumerate(tokens):
            for target_id, match_kinds in self.match(
                surface=str(token.normalized),
                lemma=str(token.lemma),
            ).items():
                matches.append(
                    TargetMatch(
                        target_id=target_id,
                        match_kinds=frozenset(match_kinds),
                        start_index=index,
                        end_index=index + 1,
                        surface=str(token.normalized),
                        lemma=str(token.lemma),
                    )
                )

        if self.max_phrase_length <= 0:
            return matches

        for start_index in range(len(tokens)):
            max_end = min(len(tokens), start_index + self.max_phrase_length)
            for end_index in range(start_index + 1, max_end + 1):
                surface_key = tuple(surfaces[start_index:end_index])
                lemma_key = tuple(lemmas[start_index:end_index])
                surface_hits = set(self.phrase_surface_to_targets.get(surface_key, set()))
                lemma_hits = set(self.phrase_lemma_to_targets.get(lemma_key, set()))
                target_ids = surface_hits | lemma_hits
                if not target_ids:
                    continue
                surface_text = " ".join(surface_key)
                lemma_text = " ".join(lemma_key)
                for target_id in target_ids:
                    matches.append(
                        TargetMatch(
                            target_id=target_id,
                            match_kinds=frozenset(
                                kind
                                for kind, hits in (
                                    ("surface", surface_hits),
                                    ("lemma", lemma_hits),
                                )
                                if target_id in hits
                            ),
                            start_index=start_index,
                            end_index=end_index,
                            surface=surface_text,
                            lemma=lemma_text,
                        )
                    )

        return sorted(
            matches,
            key=lambda match: (match.start_index, match.end_index, match.target_id),
        )


__all__ = [
    "LexicalTarget",
    "TargetMatch",
    "TargetMatcher",
    "lexeme_tokens",
    "load_targets",
    "normalize_lexeme",
    "parse_targets",
]
