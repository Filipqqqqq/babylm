"""Load and save static word-vector models for Word2Vec analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from babysteps.paths import resolve_path


def _missing_gensim_message() -> str:
    return (
        "Word2Vec analysis requires gensim. Install the optional dependency with "
        "`pip install -e '.[word2vec]'` or install gensim in your active environment."
    )


def require_gensim():
    """Import gensim lazily so tests that do not train vectors stay lightweight."""

    try:
        import gensim
    except ImportError as exc:
        raise ImportError(_missing_gensim_message()) from exc
    return gensim


def load_keyed_vectors(
    config: dict[str, Any],
    *,
    root_dir: str | Path,
):
    """Load vectors from a gensim Word2Vec model, KeyedVectors file, or binary format."""

    gensim = require_gensim()
    path = resolve_path(config["path"], base_dir=Path(root_dir))
    kind = str(config.get("kind", "keyed_vectors")).strip().lower()
    limit = config.get("limit")
    limit = None if limit in {None, ""} else int(limit)

    if kind in {"word2vec_model", "model"}:
        return gensim.models.Word2Vec.load(str(path)).wv
    if kind in {"keyed_vectors", "kv"}:
        return gensim.models.KeyedVectors.load(str(path), mmap=config.get("mmap"))
    if kind in {"word2vec_format", "word2vec_binary", "google_news"}:
        return gensim.models.KeyedVectors.load_word2vec_format(
            str(path),
            binary=bool(config.get("binary", True)),
            limit=limit,
        )
    raise ValueError(
        "vectors.kind must be one of word2vec_model, keyed_vectors, or word2vec_format."
    )


def save_keyed_vectors(vectors: Any, path: str | Path) -> Path:
    """Save gensim KeyedVectors in native ``.kv`` format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors.save(str(path))
    return path


__all__ = [
    "load_keyed_vectors",
    "require_gensim",
    "save_keyed_vectors",
]
