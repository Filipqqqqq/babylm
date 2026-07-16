#!/usr/bin/env python3
"""Train Word2Vec on generated text and evaluate EWoK context sensitivity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.word2vec.corpus import tokenization_config_from_mapping
from babysteps.analysis.word2vec.ewok_eval import evaluate_ewok_items, load_ewok_items
from babysteps.analysis.word2vec.reports import write_eval_report
from babysteps.analysis.word2vec.training import train_word2vec
from babysteps.analysis.word2vec.vectors import load_keyed_vectors
from babysteps.paths import resolve_path


DEFAULT_OUTPUT_ROOT = (
    "${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/analysis/word2vec/generated_text"
)
DEFAULT_FULL_EWOK = "external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast"
DEFAULT_MATERIAL_EVALS = [
    (
        "material_dynamics",
        "external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast/material-dynamics.jsonl",
    ),
    (
        "material_properties",
        "external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast/material-properties.jsonl",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-text",
        action="append",
        required=True,
        help="Local text file to train on. Repeat to concatenate multiple generated corpora.",
    )
    parser.add_argument(
        "--name",
        help="Run name. Defaults to the first training file stem.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for timestamped train/eval runs.",
    )
    parser.add_argument(
        "--document-mode",
        choices=("line", "file"),
        default="line",
        help="Treat each non-empty line or each file as a training document.",
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="EWoK eval path. Repeat for more domains. Defaults to full EWoK.",
    )
    parser.add_argument(
        "--no-full-ewok",
        action="store_true",
        help="Do not add the default full-EWoK eval.",
    )
    parser.add_argument(
        "--material-evals",
        action="store_true",
        help="Also evaluate material-dynamics and material-properties JSONL files.",
    )
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample", type=float, default=0.001)
    parser.add_argument("--cbow", action="store_true", help="Use CBOW instead of skip-gram.")
    parser.add_argument("--max-docs", type=int, help="Optional training document cap.")
    parser.add_argument("--max-items", type=int, help="Optional EWoK item cap per eval.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/JSON outputs only.")
    return parser.parse_args()


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("._-") or "eval"


def _parse_eval(value: str) -> tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return _safe_label(label), path
    path = value
    return _safe_label(Path(path).stem), path


def _resolve_inputs(values: list[str]) -> list[Path]:
    paths = [resolve_path(value, base_dir=ROOT) for value in values]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing training text file(s):\n{joined}")
    return paths


def _build_eval_specs(args: argparse.Namespace) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if not args.no_full_ewok:
        specs.append(("ewok_full", DEFAULT_FULL_EWOK))
    if args.material_evals:
        specs.extend(DEFAULT_MATERIAL_EVALS)
    specs.extend(_parse_eval(value) for value in args.eval)

    deduped: dict[str, str] = {}
    for label, path in specs:
        deduped[label] = path
    return list(deduped.items())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _training_config(args: argparse.Namespace, input_paths: list[Path], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dataset": {
            "kind": "local_text",
            "files": [str(path) for path in input_paths],
            "text_field": "text",
            "document_mode": args.document_mode,
            "skip_empty": True,
        },
        "preprocessing": {
            "lowercase": True,
            "split_sentences": True,
            "min_token_length": 1,
        },
        "word2vec": {
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "sg": 0 if args.cbow else 1,
            "negative": args.negative,
            "epochs": args.epochs,
            "workers": args.workers,
            "seed": args.seed,
            "sample": args.sample,
            "hs": 0,
        },
        "output": {"timestamp": False},
    }


def main() -> None:
    args = parse_args()
    input_paths = _resolve_inputs(args.train_text)
    name = _safe_label(args.name or input_paths[0].stem)
    created_utc = datetime.now(timezone.utc)
    run_dir = (
        resolve_path(args.output_root, base_dir=ROOT)
        / name
        / created_utc.strftime("%Y%m%d_%H%M%S")
    )
    train_dir = run_dir / "train"

    config = _training_config(args, input_paths, name)
    training = train_word2vec(
        config,
        root_dir=ROOT,
        output_dir=train_dir,
        max_docs=args.max_docs,
    )

    vectors = load_keyed_vectors(
        {"kind": "word2vec_model", "path": str(training.files["model"])},
        root_dir=ROOT,
    )
    tokenization = tokenization_config_from_mapping(config["preprocessing"])

    eval_summaries = []
    for label, ewok_path in _build_eval_specs(args):
        items = load_ewok_items(ewok_path, root_dir=ROOT)
        if args.max_items is not None:
            items = items[: args.max_items]
        records = evaluate_ewok_items(items, vectors, tokenization=tokenization)
        output_dir = run_dir / f"eval_{label}"
        paths = write_eval_report(
            records,
            output_dir,
            manifest={
                "name": f"{name}_{label}_word2vec",
                "created_utc": created_utc.isoformat(),
                "ewok_path": ewok_path,
                "training_model_path": str(training.files["model"]),
                "preprocessing": config["preprocessing"],
            },
            make_plots=not args.no_plots,
        )
        counted = sum(1 for record in records if record.is_counted)
        eval_summaries.append(
            {
                "label": label,
                "ewok_path": ewok_path,
                "items_total": len(records),
                "items_counted": counted,
                "output_dir": str(output_dir),
                "paths": {key: str(path) for key, path in paths.items()},
            }
        )

    run_manifest = {
        "name": name,
        "created_utc": created_utc.isoformat(),
        "run_dir": str(run_dir),
        "train_text": [str(path) for path in input_paths],
        "training": {
            "output_dir": str(training.output_dir),
            "files": {key: str(path) for key, path in training.files.items()},
            "manifest": training.manifest,
        },
        "evals": eval_summaries,
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    print(f"Run: {run_dir}")
    print(f"Model: {training.files['model']}")
    print(f"Vocab size: {training.manifest['vocab_size']:,}")
    print(f"Corpus words: {training.manifest['corpus_total_words']:,}")
    for summary in eval_summaries:
        print(
            f"Eval {summary['label']}: "
            f"{summary['items_counted']:,}/{summary['items_total']:,} counted -> "
            f"{summary['output_dir']}"
        )
    print(f"Manifest: {run_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
