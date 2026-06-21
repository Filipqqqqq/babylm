#!/usr/bin/env python3
"""Build lexical target-coverage and tokenizer-fertility reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from babysteps.analysis.lexical.pipeline import (  # noqa: E402
    load_analysis_config,
    run_lexical_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "analysis" / "lexical_babylm2026_strict.yaml"),
        help="Path to a lexical analysis YAML config.",
    )
    parser.add_argument("--output-dir", help="Override output.dir and skip timestamp nesting.")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional document cap.")
    parser.add_argument("--tokenizer-path", help="Override tokenizer.path.")
    parser.add_argument("--targets", help="Override targets.path.")
    parser.add_argument("--lemmatizer-kind", choices=["spacy", "regex"], help="Override lemmatizer.kind.")
    parser.add_argument("--lemmatizer-model", help="Override lemmatizer.model.")
    parser.add_argument("--spacy-batch-size", type=int, help="Override lemmatizer.batch_size.")
    parser.add_argument("--spacy-n-process", type=int, help="Override lemmatizer.n_process.")
    parser.add_argument(
        "--context-word-radius",
        type=int,
        help="Override analysis.context_word_radius for sampled examples.",
    )
    parser.add_argument(
        "--include-context-stopwords",
        action="store_true",
        help=(
            "Do not filter common English stopwords from the content-neighbor "
            "columns in context_summary.csv."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable lexical analysis progress logging.",
    )
    parser.add_argument(
        "--progress-interval-docs",
        type=int,
        help="Override analysis.progress_interval_docs.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        help="Override analysis.progress_interval_seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_analysis_config(args.config)

    if args.tokenizer_path:
        config.setdefault("tokenizer", {})["path"] = args.tokenizer_path
    if args.targets:
        config.setdefault("targets", {})["path"] = args.targets
    if args.lemmatizer_kind:
        config.setdefault("lemmatizer", {})["kind"] = args.lemmatizer_kind
    if args.lemmatizer_model:
        config.setdefault("lemmatizer", {})["model"] = args.lemmatizer_model
    if args.spacy_batch_size:
        config.setdefault("lemmatizer", {})["batch_size"] = int(args.spacy_batch_size)
    if args.spacy_n_process:
        config.setdefault("lemmatizer", {})["n_process"] = int(args.spacy_n_process)
    if args.context_word_radius is not None:
        config.setdefault("analysis", {})["context_word_radius"] = int(args.context_word_radius)
    if args.include_context_stopwords:
        config.setdefault("analysis", {})["filter_context_neighbor_stopwords"] = False
    if args.no_progress:
        config.setdefault("analysis", {})["progress_logging"] = False
    if args.progress_interval_docs is not None:
        config.setdefault("analysis", {})["progress_interval_docs"] = int(
            args.progress_interval_docs
        )
    if args.progress_interval_seconds is not None:
        config.setdefault("analysis", {})["progress_interval_seconds"] = float(
            args.progress_interval_seconds
        )

    result = run_lexical_analysis(
        config,
        root_dir=ROOT,
        output_dir=args.output_dir,
        max_docs=args.max_docs if args.max_docs > 0 else None,
    )

    print(f"Output dir: {result.output_dir}")
    print(f"Docs processed: {result.manifest['docs_processed']}")
    print(f"Word tokens: {result.manifest['word_tokens']}")
    print(f"Targets: {result.manifest['target_count']}")
    for name, path in result.files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
