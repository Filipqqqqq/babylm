"""Target-list parsing and matching for lexical analysis reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _normalized_values(values: Any, *, target_id: str, field: str) -> tuple[str, ...]:
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
        _validate_single_token(item, target_id=target_id, field=field)
        if item:
            normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def parse_targets(payload: dict[str, Any]) -> list[LexicalTarget]:
    """Parse and validate a v1 target YAML payload."""

    version = int(payload.get("version", 1))
    if version != 1:
        raise ValueError(f"Unsupported lexical target version {version}; expected 1.")

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

        label = str(raw_target.get("label", target_id)).strip() or target_id
        surfaces = _normalized_values(
            raw_target.get("surfaces", [label]),
            target_id=target_id,
            field="surfaces",
        )
        lemmas = _normalized_values(
            raw_target.get("lemmas", []),
            target_id=target_id,
            field="lemmas",
        )
        if not surfaces and not lemmas:
            raise ValueError(
                f"Target {target_id!r} must define at least one surface or lemma entry."
            )

        targets.append(
            LexicalTarget(
                id=target_id,
                label=label,
                surfaces=surfaces,
                lemmas=lemmas,
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
        for target in targets:
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


__all__ = [
    "LexicalTarget",
    "TargetMatcher",
    "load_targets",
    "normalize_lexeme",
    "parse_targets",
]
