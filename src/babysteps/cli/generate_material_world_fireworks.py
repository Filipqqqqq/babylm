#!/usr/bin/env python3
"""Run the material-world synthetic data probe with a Fireworks chat model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from babysteps.material_world.fireworks_client import (
    DEFAULT_FIREWORKS_CHAT_URL,
    FireworksChatClient,
    load_api_key,
)
from babysteps.material_world.fireworks_pipeline import (
    DEFAULT_PIPELINE_CONFIG_PATH,
    DeterministicMaterialWorldMockGenerator,
    load_pipeline_config,
    run_fireworks_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_PIPELINE_CONFIG_PATH),
        help="Path to configs/material_world/fireworks_probe.json.",
    )
    parser.add_argument(
        "--model",
        help="Fireworks model id, e.g. accounts/fireworks/models/...",
    )
    parser.add_argument("--api-key", help="Fireworks API key. Prefer FIREWORKS_API_KEY or --api-key-file.")
    parser.add_argument("--api-key-file", help="Plain-text file containing the Fireworks API key.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--material-class-limit", type=int, help="Number of material classes for the probe.")
    parser.add_argument("--object-limit", type=int, help="Number of object cards for the probe.")
    parser.add_argument("--style-limit", type=int, help="Number of discourse styles for the probe.")
    parser.add_argument("--scenario-limit", type=int, help="Maximum number of scenario plans/examples.")
    parser.add_argument("--audit-edit-limit", type=int, help="Maximum number of row-level audit edits.")
    parser.add_argument(
        "--seed",
        "--selection-seed",
        dest="selection_seed",
        help="Deterministically shuffle selected material/object/style catalogues before prompting.",
    )
    parser.add_argument(
        "--coverage-set",
        action="append",
        dest="coverage_sets",
        help="Optional coverage set id to target, e.g. ewok_material_properties. Can be repeated.",
    )
    parser.add_argument(
        "--object-sampling-mode",
        choices=["model_choice", "balanced_round_robin"],
        help="How scenario planning should assign objects. Use balanced_round_robin for equal selected-object exposure.",
    )
    parser.add_argument(
        "--coverage-object-fraction",
        type=float,
        help="Fraction of scenario plans assigned to use objects from each --coverage-set.",
    )
    parser.add_argument(
        "--coverage-material-fraction",
        type=float,
        help="Fraction of scenario plans assigned to use materials from each --coverage-set.",
    )
    parser.add_argument(
        "--coverage-row-fraction",
        type=float,
        help="Fraction of scenario plans assigned to each --coverage-set as general concept rows.",
    )
    parser.add_argument(
        "--coverage-dynamic-fraction",
        type=float,
        help="Fraction of scenario plans assigned to use dynamic concepts from each --coverage-set.",
    )
    parser.add_argument("--scenario-batch-size", type=int, help="Scenario plans to request per Fireworks call.")
    parser.add_argument("--render-batch-size", type=int, help="Scenario plans to render per Fireworks call.")
    parser.add_argument("--check-batch-size", type=int, help="Draft prose rows to check/edit per Fireworks call.")
    parser.add_argument("--audit-edit-batch-size", type=int, help="Flagged rows to edit per Fireworks call.")
    parser.add_argument(
        "--mock-fireworks",
        action="store_true",
        help="Use a deterministic local fake generator instead of making network/API calls.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable stage progress output.",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    for attr, key in [
        ("material_class_limit", "material_class_limit"),
        ("object_limit", "object_limit"),
        ("style_limit", "style_limit"),
        ("scenario_limit", "scenario_limit"),
        ("audit_edit_limit", "audit_edit_limit"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.selection_seed is not None:
        config["selection_seed"] = args.selection_seed
    if args.object_sampling_mode is not None:
        object_sampling = dict(config.get("object_sampling", {}))
        object_sampling["mode"] = args.object_sampling_mode
        object_sampling.setdefault("scope", "selected_objects")
        config["object_sampling"] = object_sampling
    if args.coverage_sets:
        config["coverage_set_ids"] = args.coverage_sets
    coverage_fraction_args = [
        ("coverage_row_fraction", "row_fraction"),
        ("coverage_object_fraction", "object_fraction"),
        ("coverage_material_fraction", "material_fraction"),
        ("coverage_dynamic_fraction", "dynamic_fraction"),
    ]
    if any(getattr(args, attr) is not None for attr, _ in coverage_fraction_args):
        if not args.coverage_sets:
            raise SystemExit("Pass --coverage-set when using coverage fraction options.")
        coverage_mix = dict(config.get("coverage_mix", {}))
        for coverage_set_id in args.coverage_sets:
            set_mix = dict(coverage_mix.get(coverage_set_id, {}))
            for attr, key in coverage_fraction_args:
                value = getattr(args, attr)
                if value is not None:
                    set_mix[key] = value
            coverage_mix[coverage_set_id] = set_mix
        config["coverage_mix"] = coverage_mix
    batching = dict(config.get("batching", {}))
    for attr, key in [
        ("scenario_batch_size", "scenario_plans_per_request"),
        ("render_batch_size", "render_examples_per_request"),
        ("check_batch_size", "check_examples_per_request"),
        ("audit_edit_batch_size", "audit_edit_examples_per_request"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            batching[key] = value
    if batching:
        config["batching"] = batching
    return config


def _resolve_model_id(config: dict[str, Any], args: argparse.Namespace) -> str:
    model_id = args.model or os.environ.get("FIREWORKS_MODEL_ID") or config.get("model_id")
    if model_id and str(model_id).strip():
        return str(model_id).strip()
    if args.mock_fireworks:
        return "mock-fireworks"
    raise SystemExit("Missing Fireworks model id. Pass --model or set FIREWORKS_MODEL_ID.")


def _request_settings(config: dict[str, Any]) -> dict[str, Any]:
    request = config.get("request", {})
    return {
        "url": config.get("fireworks_url", DEFAULT_FIREWORKS_CHAT_URL),
        "timeout_seconds": int(request.get("timeout_seconds", 600)),
        "max_request_retries": int(request.get("max_request_retries", 4)),
        "initial_backoff_seconds": float(request.get("initial_backoff_seconds", 1.0)),
        "max_backoff_seconds": float(request.get("max_backoff_seconds", 16.0)),
        "reasoning_effort": config.get("reasoning_effort", "none"),
        "reasoning_history": config.get("reasoning_history", "disabled"),
    }


class _CliProgress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._bar = None
        if not enabled:
            return
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None and sys.stderr.isatty():
            self._bar = tqdm(total=8, desc="material-world", unit="stage")

    def __call__(self, message: str, *, advance: bool = False) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.write(f"[material-world] {message}")
            if advance:
                self._bar.update(1)
        else:
            print(f"[material-world] {message}", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _apply_cli_overrides(load_pipeline_config(config_path), args)
    model_id = _resolve_model_id(config, args)

    if args.mock_fireworks:
        generator = DeterministicMaterialWorldMockGenerator()
    else:
        try:
            api_key = load_api_key(api_key=args.api_key, api_key_file=args.api_key_file)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        settings = _request_settings(config)
        generator = FireworksChatClient(
            model=model_id,
            api_key=api_key,
            url=settings["url"],
            timeout_seconds=settings["timeout_seconds"],
            max_request_retries=settings["max_request_retries"],
            initial_backoff_seconds=settings["initial_backoff_seconds"],
            max_backoff_seconds=settings["max_backoff_seconds"],
            reasoning_effort=settings["reasoning_effort"],
            reasoning_history=settings["reasoning_history"],
        )

    progress = _CliProgress(enabled=not args.quiet)
    try:
        manifest = run_fireworks_pipeline(
            config=config,
            generator=generator,
            model_id=model_id,
            output_dir=args.output_dir,
            config_path=config_path,
            progress=progress,
        )
    finally:
        progress.close()
    print(f"Output directory: {manifest['output_dir']}")
    print(f"Scenario plans: {manifest['counts']['scenario_plans']}")
    print(f"Final train examples: {manifest['counts']['final_train_examples']}")
    print(f"Final prose: {manifest['paths']['train_audited']}")
    print(f"Manifest: {manifest['paths']['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
