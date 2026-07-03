#!/usr/bin/env python3
"""Re-run deterministic audit checks for concept-curriculum documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from babysteps.concept_curriculum.audit import audit_documents
from babysteps.concept_curriculum.pipeline import (
    DEFAULT_CONFIG_PATH,
    build_audit_report,
    export_train_text,
    load_pipeline_config,
)


ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, *, base_dir: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object in {path}.")
                rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_output_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(str(config["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return output_dir


def active_profile(config: dict[str, Any]) -> dict[str, Any]:
    profile = (
        config.get("concept_profile")
        or config.get("material_class_profile")
        or config.get("physical_dynamics_profile")
    )
    if not isinstance(profile, dict):
        raise ValueError(
            "Config must include concept_profile, material_class_profile, or "
            "physical_dynamics_profile."
        )
    return dict(profile)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", help="Input documents JSONL. Defaults to documents_draft.jsonl.")
    parser.add_argument("--scenario-plans", help="Scenario plans JSONL. Defaults to scenario_plans.jsonl.")
    parser.add_argument("--output", help="Audited output JSONL. Defaults to documents_reaudited.jsonl.")
    parser.add_argument("--report", help="Audit report JSON. Defaults to audit_report_reaudited.json.")
    parser.add_argument("--train-output", help="Optional train text export. Defaults to train_reaudited.txt.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Write back to documents_audited.jsonl, audit_report.json, and train.txt.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    output_dir = default_output_dir(config)

    input_path = resolve_path(args.input, base_dir=Path.cwd()) if args.input else output_dir / "documents_draft.jsonl"
    scenario_path = (
        resolve_path(args.scenario_plans, base_dir=Path.cwd())
        if args.scenario_plans
        else output_dir / "scenario_plans.jsonl"
    )

    if args.replace:
        audited_path = output_dir / "documents_audited.jsonl"
        report_path = output_dir / "audit_report.json"
        train_path = output_dir / "train.txt"
    else:
        audited_path = (
            resolve_path(args.output, base_dir=Path.cwd())
            if args.output
            else output_dir / "documents_reaudited.jsonl"
        )
        report_path = (
            resolve_path(args.report, base_dir=Path.cwd())
            if args.report
            else output_dir / "audit_report_reaudited.json"
        )
        train_path = (
            resolve_path(args.train_output, base_dir=Path.cwd())
            if args.train_output
            else output_dir / "train_reaudited.txt"
        )

    specs = read_jsonl(scenario_path)
    documents = read_jsonl(input_path)
    specs_by_id = {str(spec["id"]): spec for spec in specs}

    audited = audit_documents(
        documents,
        specs_by_id=specs_by_id,
        concept_profile=active_profile(config),
        constraints=dict(config.get("document_constraints", {})),
    )
    report = build_audit_report(audited)

    write_jsonl(audited, audited_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    train_count = export_train_text(audited, train_path)

    print(f"Audited documents: {len(audited)}")
    print(f"Train documents: {train_count}")
    print(f"Counts: {report['counts']}")
    print(f"Flags: {report['flag_counts']}")
    print(f"Audited JSONL: {audited_path}")
    print(f"Audit report: {report_path}")
    print(f"Train text: {train_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
