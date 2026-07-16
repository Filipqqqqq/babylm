"""Training orchestration for BabyLM Word2Vec models."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from babysteps.analysis.word2vec.corpus import corpus_from_config
from babysteps.analysis.word2vec.vectors import require_gensim
from babysteps.paths import expand_env_vars, resolve_path


DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()


@dataclass(frozen=True)
class Word2VecTrainingResult:
    """Paths and summary metadata for a trained Word2Vec model."""

    output_dir: Path
    files: dict[str, Path]
    manifest: dict[str, Any]


def load_word2vec_config(path: str | Path) -> dict[str, Any]:
    """Load a Word2Vec YAML config with environment expansion."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def _resolve_output_dir(
    config: dict[str, Any],
    *,
    root_dir: Path,
    output_dir: str | Path | None,
    created_utc: datetime,
) -> Path:
    if output_dir is not None:
        return resolve_path(output_dir, base_dir=Path.cwd())
    raw_output = config.get("output", {})
    output_config = raw_output if isinstance(raw_output, dict) else {}
    configured = output_config.get("dir")
    if configured is None:
        configured = DEFAULT_RUNS_DIR / "analysis" / "word2vec" / str(config.get("name", "run"))
    base_dir = resolve_path(configured, base_dir=root_dir)
    if not bool(output_config.get("timestamp", True)):
        return base_dir
    return base_dir / created_utc.strftime("%Y%m%d_%H%M%S")


def _word2vec_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    w2v_config = config.get("word2vec", {})
    return {
        "vector_size": int(w2v_config.get("vector_size", 300)),
        "window": int(w2v_config.get("window", 5)),
        "min_count": int(w2v_config.get("min_count", 5)),
        "sg": int(w2v_config.get("sg", 1)),
        "negative": int(w2v_config.get("negative", 10)),
        "epochs": int(w2v_config.get("epochs", 5)),
        "workers": int(w2v_config.get("workers", 4)),
        "seed": int(w2v_config.get("seed", 13)),
        "sample": float(w2v_config.get("sample", 1e-3)),
        "hs": int(w2v_config.get("hs", 0)),
        "compute_loss": bool(w2v_config.get("compute_loss", False)),
    }


def _word2vec_progress_callbacks(gensim: Any, *, epochs: int, enabled: bool) -> list[Any]:
    if not enabled:
        return []

    from tqdm.auto import tqdm

    class TqdmEpochLogger(gensim.models.callbacks.CallbackAny2Vec):
        def __init__(self, total_epochs: int) -> None:
            self.total_epochs = total_epochs
            self.completed_epochs = 0
            self.previous_loss = 0.0
            self.progress = tqdm(
                total=total_epochs,
                desc="word2vec epochs",
                unit="epoch",
                dynamic_ncols=True,
            )

        def on_epoch_end(self, model: Any) -> None:
            self.completed_epochs += 1
            postfix = {"done": f"{self.completed_epochs}/{self.total_epochs}"}
            if getattr(model, "compute_loss", False):
                loss = float(model.get_latest_training_loss())
                postfix["loss_delta"] = f"{loss - self.previous_loss:.2f}"
                self.previous_loss = loss
            self.progress.update(1)
            self.progress.set_postfix(postfix)
            tqdm.write(f"word2vec epoch {self.completed_epochs}/{self.total_epochs} complete")

        def on_train_end(self, model: Any) -> None:
            self.progress.close()

    return [TqdmEpochLogger(epochs)]


def train_word2vec(
    config: dict[str, Any],
    *,
    root_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_docs: int | None = None,
    workers: int | None = None,
    epochs: int | None = None,
    progress: bool = True,
) -> Word2VecTrainingResult:
    """Train a gensim Word2Vec model from a restartable dataset stream."""

    gensim = require_gensim()
    root = Path(root_dir) if root_dir is not None else Path.cwd()
    created_utc = datetime.now(timezone.utc)
    resolved_output = _resolve_output_dir(
        config,
        root_dir=root,
        output_dir=output_dir,
        created_utc=created_utc,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    kwargs = _word2vec_kwargs(config)
    if workers is not None:
        kwargs["workers"] = int(workers)
    if epochs is not None:
        kwargs["epochs"] = int(epochs)

    corpus = corpus_from_config(config, max_docs=max_docs)
    epoch_count = int(kwargs["epochs"])
    if progress:
        from tqdm.auto import tqdm

        tqdm.write("word2vec building vocabulary...")
    model = gensim.models.Word2Vec(**kwargs)
    model.build_vocab(corpus)
    if progress:
        from tqdm.auto import tqdm

        tqdm.write(
            "word2vec vocabulary ready: "
            f"{len(model.wv):,} types, "
            f"{int(getattr(model, 'corpus_total_words', 0)):,} corpus words, "
            f"{epoch_count} epochs, {int(kwargs['workers'])} workers"
        )
    model.train(
        corpus,
        total_examples=int(getattr(model, "corpus_count", 0)),
        epochs=epoch_count,
        callbacks=_word2vec_progress_callbacks(gensim, epochs=epoch_count, enabled=progress),
    )

    model_path = resolved_output / "model.model"
    vectors_path = resolved_output / "vectors.kv"
    vocab_path = resolved_output / "vocab.csv"
    manifest_path = resolved_output / "manifest.json"

    model.save(str(model_path))
    model.wv.save(str(vectors_path))
    _write_vocab(model, vocab_path)

    manifest = {
        "name": config.get("name", "word2vec"),
        "created_utc": created_utc.isoformat(),
        "dataset": config.get("dataset", {}),
        "preprocessing": config.get("preprocessing", {}),
        "word2vec": kwargs,
        "max_docs": max_docs,
        "corpus_count": int(getattr(model, "corpus_count", 0)),
        "corpus_total_words": int(getattr(model, "corpus_total_words", 0)),
        "vocab_size": len(model.wv),
        "model_path": str(model_path),
        "vectors_path": str(vectors_path),
        "vocab_path": str(vocab_path),
    }
    _write_json(manifest, manifest_path)

    return Word2VecTrainingResult(
        output_dir=resolved_output,
        files={
            "model": model_path,
            "vectors": vectors_path,
            "vocab": vocab_path,
            "manifest": manifest_path,
        },
        manifest=manifest,
    )


def _write_vocab(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for word, vocab in model.wv.key_to_index.items():
        rows.append(
            {
                "word": word,
                "index": vocab,
                "count": model.wv.get_vecattr(word, "count"),
            }
        )
    rows.sort(key=lambda row: (-int(row["count"]), int(row["index"])))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "index", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


__all__ = [
    "Word2VecTrainingResult",
    "load_word2vec_config",
    "train_word2vec",
]
