"""EWoK context-sensitivity evaluation with static Word2Vec vectors."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from babysteps.analysis.word2vec.corpus import TokenizationConfig, tokenize_text
from babysteps.paths import resolve_path


REQUIRED_EWOK_FIELDS = ("Context1", "Context2", "Target1", "Target2")


@dataclass(frozen=True)
class EwokItem:
    """One EWoK two-context/two-target item."""

    item_id: str
    domain: str
    context_1: str
    target_1: str
    context_2: str
    target_2: str
    concept_a: str = ""
    concept_b: str = ""
    context_type: str = ""
    context_diff: str = ""
    target_diff: str = ""
    source_file: str = ""
    line_number: int = 0


@dataclass(frozen=True)
class PooledVector:
    """Mean-pooled vector and token coverage diagnostics for one text span."""

    vector: np.ndarray | None
    tokens: list[str]
    in_vocab_tokens: list[str]
    oov_tokens: list[str]

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    @property
    def in_vocab_count(self) -> int:
        return len(self.in_vocab_tokens)

    @property
    def oov_count(self) -> int:
        return len(self.oov_tokens)

    @property
    def retained_fraction(self) -> float:
        if not self.tokens:
            return 0.0
        return self.in_vocab_count / self.total_tokens


@dataclass(frozen=True)
class EwokScoreRecord:
    """Raw cosine scores and paper-style EWoK context-sensitivity fields."""

    item: EwokItem
    s11: float
    s21: float
    s22: float
    s12: float
    k1: float
    k2: float
    acc1: float
    acc2: float
    context_sensitivity: float
    status: str
    c1: PooledVector
    t1: PooledVector
    c2: PooledVector
    t2: PooledVector

    @property
    def is_counted(self) -> bool:
        return self.status == "counted"


def _json_rows_from_file(path: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}.")
            yield path.name, line_number, row


def _json_rows_from_zip(path: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".jsonl") and "/ewok_fast/" in name
        )
        if not names:
            names = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
        for name in names:
            with archive.open(name) as f:
                for line_number, raw_line in enumerate(f, start=1):
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"Expected JSON object at {name}:{line_number}.")
                    yield Path(name).name, line_number, row


def _iter_ewok_rows(path: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No EWoK JSONL files found under {path}")
        for file_path in files:
            yield from _json_rows_from_file(file_path)
        return

    if path.suffix == ".zip":
        yield from _json_rows_from_zip(path)
        return

    if path.suffix == ".jsonl":
        yield from _json_rows_from_file(path)
        return

    raise ValueError(f"EWoK path must be a directory, .zip, or .jsonl file: {path}")


def _require_ewok_fields(row: dict[str, Any], *, source_file: str, line_number: int) -> None:
    missing = [field for field in REQUIRED_EWOK_FIELDS if field not in row]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"EWoK row is missing {joined} at {source_file}:{line_number}")


def load_ewok_items(path: str | Path, *, root_dir: str | Path | None = None) -> list[EwokItem]:
    """Load EWoK items from an extracted directory, the fast-eval zip, or one JSONL."""

    base_dir = Path(root_dir) if root_dir is not None else Path.cwd()
    ewok_path = resolve_path(path, base_dir=base_dir)
    items: list[EwokItem] = []
    for source_file, line_number, row in _iter_ewok_rows(ewok_path):
        _require_ewok_fields(row, source_file=source_file, line_number=line_number)
        domain = str(row.get("Domain") or Path(source_file).stem)
        item_id = str(row.get("id") or row.get("UID") or f"{Path(source_file).stem}:{line_number}")
        items.append(
            EwokItem(
                item_id=item_id,
                domain=domain,
                context_1=str(row["Context1"]),
                target_1=str(row["Target1"]),
                context_2=str(row["Context2"]),
                target_2=str(row["Target2"]),
                concept_a=str(row.get("ConceptA", "")),
                concept_b=str(row.get("ConceptB", "")),
                context_type=str(row.get("ContextType", "")),
                context_diff=str(row.get("ContextDiff", "")),
                target_diff=str(row.get("TargetDiff", "")),
                source_file=source_file,
                line_number=line_number,
            )
        )
    return items


def _pooled_vector(
    text: str,
    vectors: Any,
    *,
    tokenization: TokenizationConfig,
) -> PooledVector:
    tokens = tokenize_text(text, tokenization)
    in_vocab: list[str] = []
    oov: list[str] = []
    token_vectors: list[np.ndarray] = []
    for token in tokens:
        if token in vectors:
            in_vocab.append(token)
            token_vectors.append(np.asarray(vectors[token], dtype=np.float64))
        else:
            oov.append(token)

    if not token_vectors:
        return PooledVector(vector=None, tokens=tokens, in_vocab_tokens=in_vocab, oov_tokens=oov)
    return PooledVector(
        vector=np.mean(token_vectors, axis=0),
        tokens=tokens,
        in_vocab_tokens=in_vocab,
        oov_tokens=oov,
    )


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return math.nan
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(left, right) / denominator)


def _margin(left: float, right: float) -> float:
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return float(left - right)


def _half_credit_accuracy_from_margin(margin: float) -> float:
    if math.isnan(margin):
        return math.nan
    if margin > 0:
        return 1.0
    if margin < 0:
        return 0.0
    return 0.5


def _paper_context_sensitivity_combined(acc1: float, acc2: float) -> float:
    if math.isnan(acc1) or math.isnan(acc2):
        return math.nan
    return 1.0 if acc1 == acc2 else 0.0


def _status_for_spans(spans: dict[str, PooledVector]) -> str:
    missing = [name for name, pooled in spans.items() if pooled.vector is None]
    if not missing:
        return "counted"
    return "insufficient_vectors:" + ",".join(missing)


def _build_item_record(
    item: EwokItem,
    vectors: Any,
    *,
    tokenization: TokenizationConfig,
) -> EwokScoreRecord:
    c1 = _pooled_vector(item.context_1, vectors, tokenization=tokenization)
    t1 = _pooled_vector(item.target_1, vectors, tokenization=tokenization)
    c2 = _pooled_vector(item.context_2, vectors, tokenization=tokenization)
    t2 = _pooled_vector(item.target_2, vectors, tokenization=tokenization)

    s11 = _cosine_similarity(c1.vector, t1.vector)
    s21 = _cosine_similarity(c2.vector, t1.vector)
    s22 = _cosine_similarity(c2.vector, t2.vector)
    s12 = _cosine_similarity(c1.vector, t2.vector)
    k1 = _margin(s11, s21)
    k2 = _margin(s22, s12)
    acc1 = _half_credit_accuracy_from_margin(k1)
    acc2 = _half_credit_accuracy_from_margin(k2)
    context_sensitivity = _paper_context_sensitivity_combined(acc1, acc2)

    return EwokScoreRecord(
        item=item,
        s11=s11,
        s21=s21,
        s22=s22,
        s12=s12,
        k1=k1,
        k2=k2,
        acc1=acc1,
        acc2=acc2,
        context_sensitivity=context_sensitivity,
        status=_status_for_spans({"c1": c1, "t1": t1, "c2": c2, "t2": t2}),
        c1=c1,
        t1=t1,
        c2=c2,
        t2=t2,
    )


def evaluate_ewok_items(
    items: Iterable[EwokItem],
    vectors: Any,
    *,
    tokenization: TokenizationConfig | None = None,
) -> list[EwokScoreRecord]:
    """Evaluate EWoK items using the paper-style context-sensitivity score."""

    active_tokenization = tokenization or TokenizationConfig()
    return [
        _build_item_record(item, vectors, tokenization=active_tokenization)
        for item in items
    ]


__all__ = [
    "EwokItem",
    "EwokScoreRecord",
    "PooledVector",
    "_build_item_record",
    "_cosine_similarity",
    "_half_credit_accuracy_from_margin",
    "_paper_context_sensitivity_combined",
    "_pooled_vector",
    "evaluate_ewok_items",
    "load_ewok_items",
]
