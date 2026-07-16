#!/usr/bin/env python3
"""Build a reproducible local-text mix for Word2Vec intervention probes."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.analysis.word2vec.corpus import TokenizationConfig, tokenize_text
from babysteps.paths import resolve_path


DEFAULT_PHYSICAL_EWOK = (
    "external/babylm-eval/strict/evaluation_data/fast_eval/ewok_fast/"
    "physical-dynamics.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument(
        "--base-dir",
        help="Directory of base BabyLM text files.",
    )
    base.add_argument(
        "--base-text",
        action="append",
        help="Base text file. Repeat for multiple files.",
    )
    parser.add_argument(
        "--base-glob",
        default="*",
        help="Glob used with --base-dir. Defaults to all files in the directory.",
    )
    parser.add_argument(
        "--synthetic-text",
        required=True,
        help="Generated synthetic train.txt to repeat into the mix.",
    )
    ratio = parser.add_mutually_exclusive_group(required=True)
    ratio.add_argument(
        "--synthetic-word-ratio",
        type=float,
        help=(
            "Target synthetic words divided by base words. "
            "For example, 0.01 means roughly 1 synthetic word per 100 base words."
        ),
    )
    ratio.add_argument(
        "--synthetic-target-words",
        type=int,
        help="Target number of synthetic words after repetition.",
    )
    ratio.add_argument(
        "--synthetic-repeat",
        type=int,
        help="Exact number of times to repeat the synthetic file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where synthetic_repeated.txt and manifest.json will be written.",
    )
    parser.add_argument(
        "--name",
        help="Suggested W2V run name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--eval",
        default=f"physical_dynamics={DEFAULT_PHYSICAL_EWOK}",
        help="Eval argument to include in the printed W2V command.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed to include in the printed W2V command.",
    )
    return parser.parse_args()


def _word_count_for_text(text: str) -> int:
    return len(tokenize_text(text, TokenizationConfig(lowercase=True, split_sentences=False)))


def count_words(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            count += _word_count_for_text(line)
    return count


def resolve_base_files(*, base_dir: str | None, base_text: list[str] | None, base_glob: str) -> list[Path]:
    if base_text:
        files = [resolve_path(value, base_dir=ROOT) for value in base_text]
    else:
        directory = resolve_path(str(base_dir), base_dir=ROOT)
        if not directory.is_dir():
            raise NotADirectoryError(f"Base directory does not exist: {directory}")
        files = sorted(path for path in directory.glob(base_glob) if path.is_file())

    if not files:
        raise FileNotFoundError("No base text files were found.")
    missing = [path for path in files if not path.is_file()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing base text file(s):\n{joined}")
    return files


def repeat_count_for_target(
    *,
    base_word_count: int,
    synthetic_word_count: int,
    synthetic_word_ratio: float | None,
    synthetic_target_words: int | None,
    synthetic_repeat: int | None,
) -> tuple[int, int, str]:
    if synthetic_word_count <= 0:
        raise ValueError("Synthetic text has zero words.")

    if synthetic_repeat is not None:
        if synthetic_repeat <= 0:
            raise ValueError("--synthetic-repeat must be positive.")
        return synthetic_repeat, synthetic_repeat * synthetic_word_count, "repeat"

    if synthetic_target_words is not None:
        if synthetic_target_words <= 0:
            raise ValueError("--synthetic-target-words must be positive.")
        repeat = max(1, math.ceil(synthetic_target_words / synthetic_word_count))
        return repeat, repeat * synthetic_word_count, "target_words"

    if synthetic_word_ratio is None:
        raise ValueError("One synthetic sizing option is required.")
    if synthetic_word_ratio <= 0:
        raise ValueError("--synthetic-word-ratio must be positive.")
    if base_word_count <= 0:
        raise ValueError("Base text has zero words.")
    target_words = int(math.ceil(base_word_count * synthetic_word_ratio))
    repeat = max(1, math.ceil(target_words / synthetic_word_count))
    return repeat, repeat * synthetic_word_count, "ratio"


def write_synthetic_repeated(
    *,
    synthetic_path: Path,
    output_path: Path,
    repeat_count: int,
) -> None:
    text = synthetic_path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for _ in range(repeat_count):
            f.write(text)


def build_w2v_command(
    *,
    train_texts: list[Path],
    name: str,
    eval_arg: str,
    seed: int,
) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "babylm",
        "python",
        "scripts/analysis/word2vec/run_generated_text_ewok_probe.py",
    ]
    for path in train_texts:
        command.extend(["--train-text", str(path)])
    command.extend(
        [
            "--name",
            name,
            "--output-root",
            "data/analysis/word2vec/generated_text",
            "--no-full-ewok",
            "--eval",
            eval_arg,
            "--seed",
            str(seed),
            "--epochs",
            "5",
            "--workers",
            "4",
            "--no-plots",
        ]
    )
    return command


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_text_mix(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_path(args.output_dir, base_dir=ROOT)
    synthetic_path = resolve_path(args.synthetic_text, base_dir=ROOT)
    if not synthetic_path.is_file():
        raise FileNotFoundError(f"Missing synthetic text file: {synthetic_path}")

    base_files = resolve_base_files(
        base_dir=args.base_dir,
        base_text=args.base_text,
        base_glob=args.base_glob,
    )
    base_word_counts = {str(path): count_words(path) for path in base_files}
    base_word_count = sum(base_word_counts.values())
    synthetic_word_count = count_words(synthetic_path)
    repeat_count, effective_synthetic_words, sizing_mode = repeat_count_for_target(
        base_word_count=base_word_count,
        synthetic_word_count=synthetic_word_count,
        synthetic_word_ratio=args.synthetic_word_ratio,
        synthetic_target_words=args.synthetic_target_words,
        synthetic_repeat=args.synthetic_repeat,
    )

    repeated_path = output_dir / "synthetic_repeated.txt"
    base_files_path = output_dir / "base_files.txt"
    manifest_path = output_dir / "manifest.json"
    command_path = output_dir / "w2v_command.sh"
    name = args.name or output_dir.name

    write_synthetic_repeated(
        synthetic_path=synthetic_path,
        output_path=repeated_path,
        repeat_count=repeat_count,
    )
    base_files_path.parent.mkdir(parents=True, exist_ok=True)
    base_files_path.write_text(
        "".join(f"{path}\n" for path in base_files),
        encoding="utf-8",
    )

    train_texts = base_files + [repeated_path]
    command = build_w2v_command(
        train_texts=train_texts,
        name=f"{name}_seed{args.seed}",
        eval_arg=args.eval,
        seed=args.seed,
    )
    command_text = shell_join(command)
    command_path.write_text(command_text + "\n", encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "base_files": [str(path) for path in base_files],
        "base_word_counts": base_word_counts,
        "base_word_count": base_word_count,
        "synthetic_text": str(synthetic_path),
        "synthetic_word_count": synthetic_word_count,
        "sizing_mode": sizing_mode,
        "requested_synthetic_word_ratio": args.synthetic_word_ratio,
        "requested_synthetic_target_words": args.synthetic_target_words,
        "requested_synthetic_repeat": args.synthetic_repeat,
        "repeat_count": repeat_count,
        "effective_synthetic_words": effective_synthetic_words,
        "effective_synthetic_to_base_ratio": (
            effective_synthetic_words / base_word_count if base_word_count else None
        ),
        "estimated_total_words": base_word_count + effective_synthetic_words,
        "outputs": {
            "synthetic_repeated": str(repeated_path),
            "base_files": str(base_files_path),
            "manifest": str(manifest_path),
            "w2v_command": str(command_path),
        },
        "w2v_command": command,
    }
    write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_text_mix(args)
    print(f"Base files: {len(manifest['base_files']):,}")
    print(f"Base words: {manifest['base_word_count']:,}")
    print(f"Synthetic words: {manifest['synthetic_word_count']:,}")
    print(f"Repeat count: {manifest['repeat_count']:,}")
    print(f"Effective synthetic words: {manifest['effective_synthetic_words']:,}")
    print(
        "Effective synthetic/base ratio: "
        f"{manifest['effective_synthetic_to_base_ratio']:.6f}"
    )
    print(f"Synthetic repeated: {manifest['outputs']['synthetic_repeated']}")
    print(f"Manifest: {manifest['outputs']['manifest']}")
    print("W2V command:")
    print(shell_join(manifest["w2v_command"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
