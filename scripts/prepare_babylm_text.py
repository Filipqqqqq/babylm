#!/usr/bin/env python3
"""Clean raw BabyLM text files into reproducible train/dev text folders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.data.text_cleaning import clean_text


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def clean_split(
    *,
    raw_dir: Path,
    output_dir: Path,
    suffix: str,
    sources: list[str],
    seq_length: int,
    encoding: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for source in sources:
        input_path = raw_dir / f"{source}.{suffix}"
        output_path = output_dir / input_path.name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing raw BabyLM text file: {input_path}")

        raw_text = input_path.read_text(encoding=encoding)
        if output_path.exists() and not overwrite:
            clean_chars = len(output_path.read_text(encoding=encoding))
            status = "exists"
        else:
            cleaned_text = clean_text(source, raw_text, seq_length=seq_length)
            output_path.write_text(cleaned_text, encoding=encoding)
            clean_chars = len(cleaned_text)
            status = "written"

        records.append(
            {
                "source": source,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "raw_chars": len(raw_text),
                "clean_chars": clean_chars,
                "status": status,
            }
        )
        print(f"{suffix}: {source} -> {output_path} ({status})")

    return records


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "data" / "babylm_10m_paths.yaml"),
        help="Path to a BabyLM text preparation YAML config.",
    )
    parser.add_argument("--clean-train-dir", help="Override clean.train_dir.")
    parser.add_argument("--clean-dev-dir", help="Override clean.dev_dir.")
    parser.add_argument("--manifest-path", help="Override clean.manifest_path.")
    parser.add_argument("--no-overwrite", action="store_true", help="Keep existing cleaned files.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    raw_config = config["raw"]
    clean_config = config["clean"]
    cleaning_config = config.get("cleaning", {})

    sources = list(config["sources"])
    seq_length = int(cleaning_config.get("seq_length", 128))
    encoding = cleaning_config.get("encoding", "utf-8")
    overwrite = bool(cleaning_config.get("overwrite", True)) and not args.no_overwrite

    clean_train_dir = Path(args.clean_train_dir or clean_config["train_dir"])
    clean_dev_dir = Path(args.clean_dev_dir or clean_config["dev_dir"])
    manifest_path = Path(args.manifest_path or clean_config["manifest_path"])

    train_records = clean_split(
        raw_dir=Path(raw_config["train_dir"]),
        output_dir=clean_train_dir,
        suffix="train",
        sources=sources,
        seq_length=seq_length,
        encoding=encoding,
        overwrite=overwrite,
    )
    dev_records = clean_split(
        raw_dir=Path(raw_config["dev_dir"]),
        output_dir=clean_dev_dir,
        suffix="dev",
        sources=sources,
        seq_length=seq_length,
        encoding=encoding,
        overwrite=overwrite,
    )

    manifest = {
        "name": config.get("name"),
        "config": str(Path(args.config).resolve()),
        "seq_length": seq_length,
        "encoding": encoding,
        "sources": sources,
        "splits": {
            "train": train_records,
            "dev": dev_records,
        },
    }
    save_manifest(manifest, manifest_path)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
