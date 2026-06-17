#!/usr/bin/env python3
"""Run BabyLM strict fast zero-shot eval over all checkpoints in a run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation.babylm_fast_eval import (  # noqa: E402
    DEFAULT_PIPELINE_DIR,
    EvalSettings,
    discover_checkpoints,
    evaluate_checkpoints,
    expand_env_vars,
    is_hf_checkpoint,
    resolve_path,
)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "eval" / "fast_eval.yaml"),
        help="Path to the BabyLM fast-eval config.",
    )
    parser.add_argument("--run-dir", help="Training run directory containing HF checkpoints.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        help="Explicit HF checkpoint directory. Can be passed more than once.",
    )
    parser.add_argument("--pipeline-dir", help="Override evaluation.pipeline_dir.")
    parser.add_argument("--eval-data-dir", help="Override data.eval_dir.")
    parser.add_argument("--output-dir", help="Override outputs.result_dir.")
    parser.add_argument("--backend", help="Override evaluation.backend.")
    parser.add_argument("--tasks", nargs="+", help="Tasks to run, e.g. blimp_fast ewok_fast.")
    parser.add_argument("--batch-size", type=int, help="Sentence zero-shot eval batch size.")
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 1.")
    parser.add_argument("--python", dest="python_executable", help="Python executable for eval subprocesses.")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun tasks even if reports exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--max-checkpoints", type=int, help="Evaluate only the first N discovered checkpoints.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    eval_config = config.get("evaluation", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    output_config = config.get("outputs", {})

    pipeline_dir = resolve_path(
        args.pipeline_dir or eval_config.get("pipeline_dir", DEFAULT_PIPELINE_DIR)
    )
    eval_data_dir = resolve_path(
        args.eval_data_dir
        or data_config.get("eval_dir")
        or pipeline_dir / "evaluation_data" / "fast_eval"
    )
    output_dir = resolve_path(
        args.output_dir
        or output_config.get("result_dir")
        or ROOT / "runs" / "evaluation" / eval_config.get("name", "babylm_fast_eval")
    )

    explicit_checkpoints = [resolve_path(item) for item in (args.checkpoints or [])]
    run_dir = args.run_dir or model_config.get("run_dir") or model_config.get("path")
    if explicit_checkpoints:
        checkpoints = explicit_checkpoints
    elif run_dir:
        checkpoints = discover_checkpoints(resolve_path(run_dir))
    else:
        raise SystemExit("Provide --run-dir, --checkpoint, or model.run_dir/model.path in the config.")

    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    if not checkpoints:
        raise SystemExit(f"No Hugging Face checkpoints found under: {run_dir}")
    if not args.dry_run:
        if not pipeline_dir.exists():
            raise SystemExit(f"BabyLM eval pipeline directory not found: {pipeline_dir}")
        if not eval_data_dir.exists():
            raise SystemExit(f"BabyLM fast eval data directory not found: {eval_data_dir}")
        for checkpoint in checkpoints:
            if not is_hf_checkpoint(checkpoint):
                raise SystemExit(f"Not a Hugging Face checkpoint directory: {checkpoint}")

    tasks = tuple(args.tasks or config.get("tasks") or ["blimp_fast", "reading"])
    settings = EvalSettings(
        pipeline_dir=pipeline_dir,
        eval_data_dir=eval_data_dir,
        output_dir=output_dir,
        backend=args.backend or eval_config.get("backend", "causal"),
        batch_size=args.batch_size or int(eval_config.get("batch_size", 64)),
        tasks=tasks,
        python_executable=args.python_executable or eval_config.get("python", sys.executable),
        cuda_visible_devices=args.device or eval_config.get("cuda_visible_devices"),
        revision_name=eval_config.get("revision_name"),
        skip_existing=not args.rerun_existing,
        dry_run=args.dry_run,
    )

    print(f"Pipeline: {settings.pipeline_dir}")
    print(f"Eval data: {settings.eval_data_dir}")
    print(f"Output: {settings.output_dir}")
    print(f"Backend: {settings.backend}")
    print("Checkpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint}")

    evaluate_checkpoints(checkpoints, settings)
    print(f"Summary written to: {settings.output_dir / 'summary.json'}")
    print(f"CSV written to: {settings.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
