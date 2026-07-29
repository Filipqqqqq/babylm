from __future__ import annotations

from pathlib import Path

import pytest

from babysteps.evaluation.babylm_zero_shot_eval import (
    EvalSettings,
    build_task_command,
    discover_checkpoints,
    expected_report_path,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_submission_checkpoints_are_discovered_in_training_order(tmp_path: Path) -> None:
    for name in ("final", "chck_100M", "chck_1M", "chck_10M"):
        _write_checkpoint(tmp_path / name)

    assert [path.name for path in discover_checkpoints(tmp_path)] == [
        "chck_1M",
        "chck_10M",
        "chck_100M",
        "final",
    ]


@pytest.mark.parametrize("backend", ["mntp", "causal"])
def test_submission_evaluation_uses_the_requested_backend(
    tmp_path: Path,
    backend: str,
) -> None:
    checkpoint = tmp_path / "final"
    settings = EvalSettings(
        pipeline_dir=ROOT / "external" / "babylm-eval",
        eval_data_dir=tmp_path / "evaluation_data" / "full_eval",
        output_dir=tmp_path / "outputs",
        backend=backend,
        dry_run=True,
    )

    command = build_task_command(
        checkpoint=checkpoint,
        task="blimp",
        settings=settings,
    )
    report = expected_report_path(
        output_dir=settings.output_dir,
        checkpoint=checkpoint,
        task="blimp",
        backend=backend,
    )

    assert command[command.index("--backend") + 1] == backend
    assert report.parts[-4:] == (
        backend,
        "blimp",
        "blimp_filtered",
        "best_temperature_report.txt",
    )


def test_paper_documents_the_submission_backend_split() -> None:
    readme = " ".join(
        (ROOT / "README.md").read_text(encoding="utf-8").split()
    )

    assert "GPT-BERT submissions use MNTP" in readme
    assert "Chatty BERT and GPT-2 use causal scoring" in readme
    assert (ROOT / "scripts" / "eval" / "eval_babylm_checkpoints.py").is_file()
    assert (ROOT / "configs" / "eval" / "full_eval.yaml").is_file()
