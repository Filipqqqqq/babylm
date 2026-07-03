"""Pipeline for QA-congruence linguistic repair data."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from babysteps.linguistic_repair.audit import audit_document, audit_documents
from babysteps.linguistic_repair.planning import ScenePlan, build_scene_plans
from babysteps.linguistic_repair.rendering import (
    DeterministicQaMockGenerator,
    RendererConfig,
    build_renderer,
    renderer_config_from_mapping,
)
from babysteps.paths import expand_env_vars, resolve_path


@dataclass(frozen=True)
class QaRepairResult:
    name: str
    output_dir: Path
    plans: tuple[ScenePlan, ...]
    documents: tuple[dict[str, Any], ...]
    audit_report: dict[str, Any]
    outputs: dict[str, Path]
    manifest: dict[str, Any]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f))


def build_qa_congruence_repair(
    mapping: dict[str, Any],
    *,
    base_dir: Path,
    config_path: Path | None = None,
    generator: Any | None = None,
) -> QaRepairResult:
    if not isinstance(mapping, dict):
        raise TypeError("QA repair config must be a mapping.")

    name = str(mapping["name"])
    output_dir = resolve_path(mapping["output_dir"], base_dir=base_dir)
    renderer_config = renderer_config_from_mapping(mapping.get("renderer"))
    plans = tuple(build_scene_plans(mapping))
    renderer_generator = generator
    if generator is None and bool(mapping.get("mock_fireworks", False)):
        renderer_generator = DeterministicQaMockGenerator()
    renderer = build_renderer(renderer_config, generator=renderer_generator)

    paths = {
        "scenario_plans": output_dir / "scenario_plans.jsonl",
        "render_checkpoints": output_dir / "render_checkpoints",
        "documents": output_dir / "documents.jsonl",
        "train": output_dir / "train.txt",
        "audit_report": output_dir / "audit_report.json",
        "manifest": output_dir / "manifest.json",
        "config": output_dir / "qa_congruence_config.yaml",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["render_checkpoints"].mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        shutil.copyfile(config_path, paths["config"])

    _write_jsonl(paths["scenario_plans"], [plan.to_dict() for plan in plans])
    rendered = _render_with_checkpoints(
        list(plans),
        renderer=renderer,
        renderer_config=renderer_config,
        checkpoint_dir=paths["render_checkpoints"],
    )
    audit = audit_documents(rendered, list(plans))
    documents = tuple(audit.pop("documents"))
    _write_jsonl(paths["documents"], list(documents))
    passing = [row for row in documents if row["audit_status"] == "pass"]
    _write_train(paths["train"], passing)
    audit_report = {
        "name": name,
        **audit,
        "train_documents": len(passing),
        "total_documents": len(documents),
    }
    _write_json(paths["audit_report"], audit_report)

    manifest = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(mapping.get("seed", 42)),
        "document_count": len(plans),
        "renderer": {
            "type": renderer_config.kind,
            "model_id": renderer_config.model_id,
            "documents_per_request": renderer_config.documents_per_request,
        },
        "counts": {
            "scenario_plans": len(plans),
            "rendered_documents": len(documents),
            "train_documents": len(passing),
            **audit_report["counts"],
        },
        "paths": {key: str(path) for key, path in paths.items()},
    }
    _write_json(paths["manifest"], manifest)
    return QaRepairResult(
        name=name,
        output_dir=output_dir,
        plans=plans,
        documents=documents,
        audit_report=audit_report,
        outputs=paths,
        manifest=manifest,
    )


def _render_with_checkpoints(
    plans: list[ScenePlan],
    *,
    renderer: Any,
    renderer_config: RendererConfig,
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    batch_size = max(1, int(renderer_config.documents_per_request))
    for start in range(0, len(plans), batch_size):
        batch = plans[start : start + batch_size]
        checkpoint_path = checkpoint_dir / f"batch_{start:06d}_n{len(batch):03d}.jsonl"
        checkpointed = (
            _load_checkpoint(checkpoint_path, batch)
            if renderer_config.resume_from_checkpoints
            else None
        )
        if checkpointed is None:
            checkpointed = renderer.render_batch(batch)
            _write_jsonl(checkpoint_path, checkpointed)
        documents.extend(checkpointed)
    return documents


def _load_checkpoint(path: Path, plans: list[ScenePlan]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    rows = _read_jsonl(path)
    expected_ids = [plan.id for plan in plans]
    if [str(row.get("id")) for row in rows] != expected_ids:
        return None
    for row, plan in zip(rows, plans):
        if audit_document(str(row.get("text", "")), plan).status != "pass":
            return None
    return rows


def _write_train(path: Path, documents: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for document in documents:
            text = str(document["text"]).strip()
            if text:
                f.write(text)
                f.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
