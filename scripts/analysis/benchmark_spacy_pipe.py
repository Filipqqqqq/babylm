#!/usr/bin/env python3
"""Benchmark spaCy nlp.pipe batch_size and n_process settings."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.data.sources import iter_dataset_rows
from babysteps.paths import expand_env_vars, resolve_path


DEFAULT_RUNS_DIR = Path(
    os.environ.get("BABYSTEPS_RUNS_DIR", "~/babylm-baby-steps-runs")
).expanduser()
DEFAULT_OUTPUT_CSV = (
    DEFAULT_RUNS_DIR / "analysis" / "spacy_pipe" / "spacy_pipe_benchmark.csv"
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return expand_env_vars(payload)


def babylm_10m_dataset_from_config(
    path: str | Path,
    *,
    split: str,
) -> dict[str, Any]:
    """Build a local_text dataset block from configs/data/babylm_10m_paths.yaml."""

    config = load_yaml(path)
    clean_config = config["clean"]
    clean_dir = resolve_path(clean_config[f"{split}_dir"], base_dir=ROOT)
    sources = list(config["sources"])
    suffix = "train" if split == "train" else "dev"
    files = [clean_dir / f"{source}.{suffix}" for source in sources]
    missing = [path for path in files if not path.exists()]
    if missing:
        preview = "\n  - ".join(str(path) for path in missing[:8])
        raise FileNotFoundError(
            "Missing cleaned BabyLM 10M files. Run:\n"
            "  python scripts/data/prepare_babylm_text.py "
            "--config configs/data/babylm_10m_paths.yaml\n"
            f"Missing:\n  - {preview}"
        )

    return {
        "kind": "local_text",
        "files": [str(path) for path in files],
        "text_field": "text",
        "document_mode": "line",
        "encoding": config.get("cleaning", {}).get("encoding", "utf-8"),
        "skip_empty": True,
    }


def collect_texts(
    dataset_config: dict[str, Any],
    *,
    text_field: str,
    max_docs: int,
    max_chars: int,
) -> list[str]:
    """Materialize a bounded benchmark sample so each setting sees the same data."""

    texts = []
    for row in iter_dataset_rows(dataset_config):
        text = str(row.get(text_field, ""))
        if not text.strip():
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        texts.append(text)
        if len(texts) >= max_docs:
            break
    if not texts:
        raise ValueError("No benchmark texts were collected from the configured dataset.")
    return texts


def count_doc_tokens(doc) -> int:
    return sum(1 for token in doc if not token.is_space)


def benchmark_setting(
    nlp,
    texts: list[str],
    *,
    batch_size: int,
    n_process: int,
    repeats: int,
    warmup_docs: int,
) -> dict[str, Any]:
    """Benchmark one batch/process setting."""

    if warmup_docs > 0:
        for _doc in nlp.pipe(
            texts[:warmup_docs],
            batch_size=batch_size,
            n_process=n_process,
        ):
            pass

    elapsed_values = []
    token_counts = []
    for _repeat in range(repeats):
        started = time.perf_counter()
        token_count = 0
        doc_count = 0
        for doc in nlp.pipe(
            texts,
            batch_size=batch_size,
            n_process=n_process,
        ):
            token_count += count_doc_tokens(doc)
            doc_count += 1
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)
        token_counts.append(token_count)

    median_elapsed = statistics.median(elapsed_values)
    median_tokens = int(statistics.median(token_counts))
    docs_per_sec = len(texts) / median_elapsed if median_elapsed else 0.0
    tokens_per_sec = median_tokens / median_elapsed if median_elapsed else 0.0
    return {
        "batch_size": batch_size,
        "n_process": n_process,
        "docs": len(texts),
        "tokens": median_tokens,
        "repeats": repeats,
        "median_seconds": f"{median_elapsed:.6f}",
        "docs_per_sec": f"{docs_per_sec:.3f}",
        "tokens_per_sec": f"{tokens_per_sec:.3f}",
        "all_seconds": " ".join(f"{value:.6f}" for value in elapsed_values),
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "n_process",
        "docs",
        "tokens",
        "repeats",
        "median_seconds",
        "docs_per_sec",
        "tokens_per_sec",
        "all_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive.")
    return values


def load_spacy_model(model: str, *, disable: list[str]):
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for this benchmark. Install with: "
            "pip install -e '.[analysis]'"
        ) from exc

    try:
        nlp = spacy.load(model, disable=disable)
    except OSError as exc:
        raise OSError(f"Could not load {model!r}. Run: python -m spacy download {model}") from exc
    return nlp


def sort_results(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["tokens_per_sec"]),
            int(row["n_process"]),
            int(row["batch_size"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--babylm-10m-config",
        default=str(ROOT / "configs" / "data" / "babylm_10m_paths.yaml"),
        help=(
            "BabyLM 10M path config. Defaults to cleaned local text files and "
            "treats each non-empty line as one benchmark document."
        ),
    )
    source.add_argument(
        "--dataset-config",
        help=(
            "Alternative YAML config containing a dataset block compatible with "
            "configs/tokenization/*_shards.yaml."
        ),
    )
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("32 64 128 256 512 1000"))
    parser.add_argument("--n-processes", type=parse_int_list, default=parse_int_list("1 2 4"))
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Optionally truncate each text to this many characters. 0 disables truncation.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-docs", type=int, default=100)
    parser.add_argument(
        "--disable",
        default="ner",
        help="Comma/space-separated spaCy pipes to disable. Default: ner.",
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_docs <= 0:
        raise ValueError("--max-docs must be positive.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    if args.dataset_config:
        config = load_yaml(args.dataset_config)
        dataset_config = config.get("dataset", config)
    else:
        dataset_config = babylm_10m_dataset_from_config(
            args.babylm_10m_config,
            split=args.split,
        )

    texts = collect_texts(
        dataset_config,
        text_field=args.text_field,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
    )
    disable = [item for item in args.disable.replace(",", " ").split() if item]
    nlp = load_spacy_model(args.model, disable=disable)

    print(f"Texts: {len(texts):,}")
    print(f"Model: {args.model}")
    print(f"Disabled pipes: {', '.join(disable) if disable else '(none)'}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"n_process values: {args.n_processes}")

    rows = []
    for batch_size, n_process in itertools.product(args.batch_sizes, args.n_processes):
        print(f"benchmarking batch_size={batch_size} n_process={n_process}", flush=True)
        row = benchmark_setting(
            nlp,
            texts,
            batch_size=batch_size,
            n_process=n_process,
            repeats=args.repeats,
            warmup_docs=min(args.warmup_docs, len(texts)),
        )
        rows.append(row)
        print(
            f"  {row['tokens_per_sec']} tokens/sec; "
            f"{row['docs_per_sec']} docs/sec; "
            f"median {row['median_seconds']} sec"
        )

    sorted_rows = sort_results(rows)
    write_csv(sorted_rows, args.output_csv)
    best = sorted_rows[0]
    print(f"CSV: {args.output_csv}")
    print(
        "Best: "
        f"batch_size={best['batch_size']} n_process={best['n_process']} "
        f"tokens/sec={best['tokens_per_sec']}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
