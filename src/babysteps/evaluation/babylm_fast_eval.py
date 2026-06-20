"""Post-training wrappers for the BabyLM strict fast evaluation pipeline."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from babysteps.paths import expand_env_vars, resolve_path as _resolve_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIPELINE_DIR = ROOT / "external" / "babylm-eval" / "strict"

HF_MODEL_FILES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}

SENTENCE_TASKS = {
    "blimp_fast": ("blimp", "blimp_fast"),
    "supplement_fast": ("blimp", "supplement_fast"),
    "ewok_fast": ("ewok", "ewok_fast"),
    "entity_tracking_fast": ("entity_tracking", "entity_tracking_fast"),
}


def resolve_path(value: str | Path, *, base_dir: Path = ROOT) -> Path:
    """Resolve a config path, treating relative paths as repo-relative."""

    return _resolve_path(value, base_dir=base_dir)


@dataclass(frozen=True)
class EvalSettings:
    pipeline_dir: Path
    eval_data_dir: Path
    output_dir: Path
    backend: str = "causal"
    batch_size: int = 64
    tasks: tuple[str, ...] = (
        "blimp_fast",
        "supplement_fast",
        "ewok_fast",
        "entity_tracking_fast",
        "reading",
    )
    python_executable: str = sys.executable
    cuda_visible_devices: str | None = None
    revision_name: str | None = None
    skip_existing: bool = True
    dry_run: bool = False


def is_hf_checkpoint(path: str | Path) -> bool:
    """Return true when a directory looks like a HF model checkpoint."""

    path = Path(path)
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return any((path / filename).exists() for filename in HF_MODEL_FILES)


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort common checkpoint names in training order, with final last."""

    name = path.name
    match = re.fullmatch(r"chck_(\d+)M", name)
    if match:
        return (0, int(match.group(1)), name)

    match = re.fullmatch(r"checkpoint-epoch-(\d+)", name)
    if match:
        return (1, int(match.group(1)), name)

    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match:
        return (2, int(match.group(1)), name)

    if name == "final":
        return (9, 0, name)

    numbers = re.findall(r"\d+", name)
    first_number = int(numbers[0]) if numbers else 0
    return (5, first_number, name)


def discover_checkpoints(run_dir: str | Path) -> list[Path]:
    """Find HF checkpoint directories below a training run directory."""

    run_dir = Path(run_dir)
    if is_hf_checkpoint(run_dir):
        return [run_dir]

    checkpoints = []
    for config_path in run_dir.rglob("config.json"):
        candidate = config_path.parent
        if is_hf_checkpoint(candidate):
            checkpoints.append(candidate)

    unique = sorted(set(checkpoints), key=checkpoint_sort_key)
    return unique


def backend_for_reading(backend: str) -> str:
    """Map sentence backends onto the reading evaluator's backend names."""

    if "enc_dec" in backend:
        return "enc_dec"
    return backend


def normalize_metric_name(name: str) -> str:
    """Normalize report labels into stable JSON/CSV metric names."""

    return name.strip().lower().replace(" ", "_").replace("-", "_")


def expected_report_path(
    *,
    output_dir: Path,
    checkpoint: Path,
    task: str,
    backend: str,
    revision_name: str | None = None,
) -> Path:
    """Return the report path that the official pipeline should write."""

    revision = revision_name or "main"
    model_name = checkpoint.stem
    base = output_dir / model_name / revision / "zero_shot"

    if task in SENTENCE_TASKS:
        official_task, dataset_name = SENTENCE_TASKS[task]
        return base / backend / official_task / dataset_name / "best_temperature_report.txt"

    if task == "reading":
        return base / backend_for_reading(backend) / "reading" / "report.txt"

    raise ValueError(f"Unsupported BabyLM fast-eval task: {task!r}")


def build_task_command(
    *,
    checkpoint: Path,
    task: str,
    settings: EvalSettings,
) -> list[str]:
    """Build the official BabyLM evaluation command for one task."""

    command = [settings.python_executable, "-m"]

    if task in SENTENCE_TASKS:
        official_task, dataset_name = SENTENCE_TASKS[task]
        command += [
            "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name",
            str(checkpoint),
            "--backend",
            settings.backend,
            "--task",
            official_task,
            "--data_path",
            str(settings.eval_data_dir / dataset_name),
            "--output_dir",
            str(settings.output_dir),
            "--save_predictions",
            "--batch_size",
            str(settings.batch_size),
        ]
    elif task == "reading":
        command += [
            "evaluation_pipeline.reading.run",
            "--model_path_or_name",
            str(checkpoint),
            "--backend",
            backend_for_reading(settings.backend),
            "--data_path",
            str(settings.eval_data_dir / "reading" / "reading_data.csv"),
            "--output_dir",
            str(settings.output_dir),
        ]
    else:
        raise ValueError(f"Unsupported BabyLM fast-eval task: {task!r}")

    if settings.revision_name:
        command += ["--revision_name", settings.revision_name]

    return command


def run_task(checkpoint: Path, task: str, settings: EvalSettings) -> None:
    """Run one official BabyLM task unless it already has a report."""

    report_path = expected_report_path(
        output_dir=settings.output_dir,
        checkpoint=checkpoint,
        task=task,
        backend=settings.backend,
        revision_name=settings.revision_name,
    )
    if settings.skip_existing and report_path.exists():
        print(f"[skip] {checkpoint.name} {task}: {report_path}")
        return

    command = build_task_command(checkpoint=checkpoint, task=task, settings=settings)
    print(f"[eval] {checkpoint.name} {task}")
    print("       " + " ".join(command))

    if settings.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if settings.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

    subprocess.run(command, cwd=settings.pipeline_dir, env=env, check=True)


def parse_best_temperature_report(path: str | Path) -> dict[str, Any]:
    """Parse a sentence-ranking report into average and section scores."""

    path = Path(path)
    result: dict[str, Any] = {
        "report_path": str(path),
        "temperature": None,
        "average": None,
        "sections": {},
    }
    current_section: str | None = None
    reading_average = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("TEMPERATURE:"):
            _, _, value = line.partition(":")
            result["temperature"] = float(value.strip())
            continue

        if line.startswith("### "):
            header = line.removeprefix("### ").strip()
            reading_average = header.startswith("AVERAGE")
            current_section = None
            if not reading_average:
                section_name = header
                if section_name.endswith(" ACCURACY"):
                    section_name = section_name[: -len(" ACCURACY")]
                current_section = section_name.lower().replace(" ", "_").replace("-", "_")
                result["sections"].setdefault(current_section, {})
            continue

        if reading_average:
            result["average"] = float(line)
            reading_average = False
            continue

        if current_section and ":" in line:
            key, _, value = line.partition(":")
            result["sections"][current_section][key.strip()] = float(value.strip())

    return result


def parse_reading_report(report_path: str | Path, correlations_path: str | Path | None = None) -> dict[str, Any]:
    """Parse the reading evaluator report and optional correlations file."""

    report_path = Path(report_path)
    scores: dict[str, float] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            scores[normalize_metric_name(key)] = float(value.strip())

    correlations = {}
    if correlations_path:
        correlations_path = Path(correlations_path)
        if correlations_path.exists():
            for line in correlations_path.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("\t")
                correlations[key.strip()] = float(value.strip())

    return {
        "report_path": str(report_path),
        "scores": scores,
        "correlations": correlations,
    }


def summarize_checkpoint(
    *,
    checkpoint: Path,
    settings: EvalSettings,
) -> dict[str, Any]:
    """Collect parsed reports for one checkpoint."""

    tasks: dict[str, Any] = {}
    for task in settings.tasks:
        report_path = expected_report_path(
            output_dir=settings.output_dir,
            checkpoint=checkpoint,
            task=task,
            backend=settings.backend,
            revision_name=settings.revision_name,
        )
        if not report_path.exists():
            tasks[task] = {"missing": True, "report_path": str(report_path)}
            continue

        if task == "reading":
            tasks[task] = parse_reading_report(
                report_path,
                report_path.parent / "correlations.txt",
            )
        else:
            tasks[task] = parse_best_temperature_report(report_path)

    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "tasks": tasks,
    }


def flatten_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten checkpoint summaries into CSV-friendly score rows."""

    rows: list[dict[str, Any]] = []
    for checkpoint in summary["checkpoints"]:
        checkpoint_name = checkpoint["checkpoint"]
        checkpoint_path = checkpoint["checkpoint_path"]
        for task, task_result in checkpoint["tasks"].items():
            if task_result.get("missing"):
                continue

            if "average" in task_result and task_result["average"] is not None:
                rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "checkpoint_path": checkpoint_path,
                        "task": task,
                        "metric_scope": "average",
                        "metric_name": "average",
                        "value": task_result["average"],
                    }
                )
                for section, values in task_result.get("sections", {}).items():
                    for metric_name, value in values.items():
                        rows.append(
                            {
                                "checkpoint": checkpoint_name,
                                "checkpoint_path": checkpoint_path,
                                "task": task,
                                "metric_scope": section,
                                "metric_name": metric_name,
                                "value": value,
                            }
                        )

            if "scores" in task_result:
                for metric_name, value in task_result["scores"].items():
                    rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "checkpoint_path": checkpoint_path,
                            "task": task,
                            "metric_scope": "report",
                            "metric_name": metric_name,
                            "value": value,
                        }
                    )
                for metric_name, value in task_result.get("correlations", {}).items():
                    rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "checkpoint_path": checkpoint_path,
                            "task": task,
                            "metric_scope": "correlation",
                            "metric_name": metric_name,
                            "value": value,
                        }
                    )

    return rows


def write_summary(summary: dict[str, Any], output_dir: str | Path) -> None:
    """Write JSON and CSV summaries for an eval run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = flatten_summary_rows(summary)
    csv_path = output_dir / "summary.csv"
    fieldnames = ["checkpoint", "checkpoint_path", "task", "metric_scope", "metric_name", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoints(checkpoints: list[Path], settings: EvalSettings) -> dict[str, Any]:
    """Run configured tasks over checkpoints and write parsed summaries."""

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        for task in settings.tasks:
            run_task(checkpoint, task, settings)
        checkpoint_summary = summarize_checkpoint(checkpoint=checkpoint, settings=settings)
        summaries.append(checkpoint_summary)
        write_summary(
            {
                "backend": settings.backend,
                "eval_data_dir": str(settings.eval_data_dir),
                "pipeline_dir": str(settings.pipeline_dir),
                "checkpoints": summaries,
            },
            settings.output_dir,
        )

    summary = {
        "backend": settings.backend,
        "eval_data_dir": str(settings.eval_data_dir),
        "pipeline_dir": str(settings.pipeline_dir),
        "checkpoints": summaries,
    }
    write_summary(summary, settings.output_dir)
    return summary
