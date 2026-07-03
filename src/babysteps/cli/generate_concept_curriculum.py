#!/usr/bin/env python3
"""Run a concept-curriculum Fireworks generation job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from babysteps.concept_curriculum.pipeline import (
    DEFAULT_CONFIG_PATH,
    CONCEPT_DOCUMENT_TYPES,
    DeterministicConceptCurriculumMockGenerator,
    load_pipeline_config,
    run_concept_curriculum_pipeline,
)
from babysteps.material_world.fireworks_client import (
    DEFAULT_FIREWORKS_CHAT_URL,
    FireworksChatClient,
    load_api_key,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a concept-curriculum JSON/YAML config.",
    )
    parser.add_argument("--model", help="Fireworks model id.")
    parser.add_argument("--api-key", help="Fireworks API key. Prefer FIREWORKS_API_KEY or --api-key-file.")
    parser.add_argument("--api-key-file", help="Plain-text file containing the Fireworks API key.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument(
        "--document-count",
        type=int,
        help=(
            "Override the total document count while preserving configured "
            "document or canonical-concept proportions."
        ),
    )
    parser.add_argument(
        "--documents-per-request",
        type=int,
        help="Override the document-render batch size for each Fireworks request.",
    )
    parser.add_argument(
        "--focus-tag",
        action="append",
        default=[],
        help="Restrict material_class_profile generation to classes with this tag. May be repeated.",
    )
    parser.add_argument(
        "--mock-fireworks",
        action="store_true",
        help="Use deterministic local generation instead of making API calls.",
    )
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


def _apply_document_count_override(config: dict[str, Any], document_count: int | None) -> dict[str, Any]:
    if document_count is None:
        return config
    if document_count <= 0:
        raise SystemExit("--document-count must be positive.")

    config = dict(config)
    packs = [dict(pack) for pack in config.get("coverage_packs", [])]
    if len(packs) != 1:
        raise SystemExit("--document-count override currently expects one coverage pack.")

    has_document_counts = bool(packs[0].get("document_counts"))
    has_canonical_counts = bool(
        packs[0].get("canonical_concept_counts")
        or packs[0].get("canonical_document_counts")
    )
    if has_document_counts and has_canonical_counts:
        raise SystemExit(
            "--document-count override expects document_counts or "
            "canonical_concept_counts, not both."
        )

    count_key = "document_counts"
    counts = dict(packs[0].get("document_counts", {}))
    ordered_keys = list(CONCEPT_DOCUMENT_TYPES) + [
        key for key in counts if key not in CONCEPT_DOCUMENT_TYPES
    ]
    if not counts:
        count_key = (
            "canonical_concept_counts"
            if packs[0].get("canonical_concept_counts")
            else "canonical_document_counts"
        )
        counts = dict(packs[0].get(count_key, {}))
        ordered_keys = list(counts)

    keys = [key for key in ordered_keys if key in counts]
    if not keys:
        raise SystemExit(
            "Config has no document_counts or canonical_concept_counts to scale."
        )

    base_total = sum(int(counts[key]) for key in keys)
    if base_total <= 0:
        raise SystemExit("Configured document counts must sum to a positive value.")
    scaled = {}
    remainder = int(document_count)
    for key in keys:
        value = int(document_count) * int(counts[key]) // base_total
        scaled[key] = value
        remainder -= value
    for key in keys[:remainder]:
        scaled[key] += 1
    packs[0][count_key] = scaled
    config["coverage_packs"] = packs
    return config


def _apply_documents_per_request_override(
    config: dict[str, Any],
    documents_per_request: int | None,
) -> dict[str, Any]:
    if documents_per_request is None:
        return config
    if documents_per_request <= 0:
        raise SystemExit("--documents-per-request must be positive.")

    config = dict(config)
    batching = dict(config.get("batching", {}))
    batching["documents_per_request"] = int(documents_per_request)
    config["batching"] = batching
    return config


def _apply_focus_tags(config: dict[str, Any], focus_tags: list[str]) -> dict[str, Any]:
    if not focus_tags:
        return config
    config = dict(config)
    existing = [str(tag) for tag in config.get("focus_tags", []) if str(tag).strip()]
    config["focus_tags"] = existing + [str(tag) for tag in focus_tags if str(tag).strip()]
    return config


class _CliProgress:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, message: str) -> None:
        if self.enabled:
            print(f"[concept-curriculum] {message}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _apply_document_count_override(load_pipeline_config(config_path), args.document_count)
    config = _apply_documents_per_request_override(config, args.documents_per_request)
    config = _apply_focus_tags(config, args.focus_tag)
    model_id = _resolve_model_id(config, args)

    if args.mock_fireworks:
        generator = DeterministicConceptCurriculumMockGenerator()
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

    manifest = run_concept_curriculum_pipeline(
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
    print(f"Train text: {manifest['paths']['train']}")
    print(f"Manifest: {manifest['paths']['manifest']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
