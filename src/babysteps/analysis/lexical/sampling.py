"""Bounded deterministic sampling utilities for lexical context examples."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16)


@dataclass
class ReservoirSampler:
    """Keep a uniform bounded sample from a stream."""

    limit: int
    seed: int
    seen: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("ReservoirSampler limit must be non-negative.")
        self._rng = random.Random(self.seed)

    def add(self, item: dict[str, Any]) -> None:
        if self.limit == 0:
            self.seen += 1
            return
        self.seen += 1
        if len(self.samples) < self.limit:
            self.samples.append(item)
            return

        index = self._rng.randrange(self.seen)
        if index < self.limit:
            self.samples[index] = item


__all__ = ["ReservoirSampler", "stable_seed"]
