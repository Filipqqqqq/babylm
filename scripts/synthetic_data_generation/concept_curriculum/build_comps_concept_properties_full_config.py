#!/usr/bin/env python3
"""Build a full COMPS concept-properties config from concept type annotations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_COMPS_PATH = (
    "external/babylm-eval/strict/evaluation_data/full_eval/comps/comps_base.jsonl"
)
DEFAULT_ANNOTATIONS = (
    "configs/concept_curriculum/comps_concept_properties/"
    "comps_concept_types_qwen_batch40.json"
)
DEFAULT_SENSE_OVERRIDES = (
    "configs/concept_curriculum/comps_concept_properties/"
    "comps_concept_sense_overrides.json"
)
DEFAULT_OUTPUT = (
    "configs/concept_curriculum/comps_concept_properties/"
    "comps_concept_properties_clean_no_neighbors_full521_qwen.json"
)
DEFAULT_NAME = "comps_concept_properties_clean_no_neighbors_full521_qwen"
DEFAULT_DATA_OUTPUT_DIR = (
    "data/concept_curriculum/comps_concept_properties_clean_no_neighbors_full521_qwen"
)


def _load_comps_module() -> Any:
    module_name = "comps_concept_properties_for_full_config"
    module_path = ROOT / "src/babysteps/concept_curriculum/comps_concept_properties.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPS = _load_comps_module()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    plans = COMPS.build_comps_concept_plans(config)
    prompts = COMPS.build_prompt_rows_from_config(config, plans)
    type_counts = Counter(
        concept_type
        for plan in plans
        for concept_type in getattr(plan, "concept_types", ())
    )
    untyped = [
        plan.id
        for plan in plans
        if not getattr(plan, "concept_types", ())
    ]
    estimated_words = sum(int(row.get("target_words", 0)) for row in prompts)
    return {
        "concepts": len(plans),
        "prompts": len(prompts),
        "estimated_words": estimated_words,
        "type_counts": dict(sorted(type_counts.items())),
        "untyped_count": len(untyped),
        "untyped": untyped,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comps-path", default=DEFAULT_COMPS_PATH)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--sense-overrides", default=DEFAULT_SENSE_OVERRIDES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--data-output-dir", default=DEFAULT_DATA_OUTPUT_DIR)
    parser.add_argument("--api-key-file", default="fireworks_api.txt")
    parser.add_argument("--model-id", default=COMPS.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--renderer-type",
        default=COMPS.FIREWORKS,
        choices=sorted(COMPS.RENDERER_TYPES),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    comps_path = resolve_repo_path(args.comps_path)
    annotations_path = resolve_repo_path(args.annotations)
    output_path = resolve_repo_path(args.output)

    concepts = COMPS.load_comps_acceptable_concepts(comps_path)
    annotations = COMPS.load_concept_type_annotations(annotations_path)
    sense_overrides = {}
    if str(args.sense_overrides).strip():
        sense_overrides_path = resolve_repo_path(args.sense_overrides)
        if sense_overrides_path.exists():
            sense_overrides = COMPS.load_concept_sense_overrides(sense_overrides_path)
        else:
            raise FileNotFoundError(f"sense overrides file not found: {sense_overrides_path}")
    config = COMPS.build_full_comps_concept_properties_config(
        concepts=concepts,
        concept_type_annotations=annotations,
        concept_sense_overrides=sense_overrides,
        name=str(args.name),
        output_dir=str(args.data_output_dir),
        renderer_type=str(args.renderer_type),
        model_id=str(args.model_id),
        api_key_file=str(args.api_key_file),
    )
    write_json(output_path, config)
    summary = summarize_config(config)
    print(
        f"wrote {output_path} concepts={summary['concepts']} "
        f"prompts={summary['prompts']} estimated_words={summary['estimated_words']} "
        f"untyped={summary['untyped_count']}"
    )
    print("type_counts=" + json.dumps(summary["type_counts"], sort_keys=True))
    if summary["untyped"]:
        print("untyped=" + ", ".join(summary["untyped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
