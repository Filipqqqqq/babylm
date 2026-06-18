#!/usr/bin/env python3
"""Run a toy EWoK-style social-properties Fireworks/Qwen generation job."""

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
from babysteps.social_properties.pipeline import (
    DEFAULT_CONFIG_PATH,
    DeterministicSocialPropertiesMockGenerator,
    load_pipeline_config,
    run_social_properties_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to a social-properties JSON/YAML config.")
    parser.add_argument("--model", help="Fireworks model id, e.g. accounts/fireworks/models/qwen3p6-plus.")
    parser.add_argument("--api-key", help="Fireworks API key. Prefer FIREWORKS_API_KEY or --api-key-file.")
    parser.add_argument("--api-key-file", help="Plain-text file containing the Fireworks API key.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--n-examples", type=int, help="Override the configured number of examples.")
    parser.add_argument("--mock-fireworks", "--dry-run", action="store_true", help="Use deterministic local generation.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output.")
    return parser.parse_args(argv)


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

    def __call__(self, message: str) -> None:
        if self.enabled:
            print(f"[social-properties] {message}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = dict(load_pipeline_config(config_path))
    if args.n_examples is not None:
        if args.n_examples <= 0:
            raise SystemExit("--n-examples must be positive.")
        config["n_examples"] = args.n_examples
    model_id = _resolve_model_id(config, args)

    if args.mock_fireworks:
        generator = DeterministicSocialPropertiesMockGenerator()
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

    manifest = run_social_properties_pipeline(
        config=config,
        generator=generator,
        model_id=model_id,
        output_dir=args.output_dir,
        config_path=config_path,
        progress=_CliProgress(enabled=not args.quiet),
    )
    print(f"Output directory: {manifest['output_dir']}")
    print(f"Scenario plans: {manifest['counts']['scenario_plans']}")
    print(f"Train documents: {manifest['counts']['train_documents']}")
    print(f"Exact trait label density: {manifest['audit_summary']['exact_trait_label_density']:.3f}")
    print(f"Train text: {manifest['paths']['train_pilot']}")
    print(f"Manifest: {manifest['paths']['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
