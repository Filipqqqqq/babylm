from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from babysteps.evaluation.babylm_zero_shot_eval import (
    EvalSettings,
    build_task_command,
    discover_checkpoints,
    expected_report_path,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPTS = (
    ROOT / "scripts" / "eval" / "setup_babylm_eval.sh",
    ROOT / "scripts" / "eval" / "eval_babylm_strict_small_submission.sh",
    ROOT / "scripts" / "eval" / "eval_babylm_strict_submission.sh",
)


def _load_script(relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}", encoding="utf-8")
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
def test_local_evaluation_uses_the_requested_backend(tmp_path: Path, backend: str) -> None:
    checkpoint = tmp_path / "final"
    settings = EvalSettings(
        pipeline_dir=ROOT / "external" / "babylm-eval",
        eval_data_dir=tmp_path / "evaluation_data" / "full_eval",
        output_dir=tmp_path / "outputs",
        backend=backend,
        dry_run=True,
    )

    command = build_task_command(checkpoint=checkpoint, task="blimp", settings=settings)
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


def test_track_wrappers_have_documented_defaults_and_backend_override() -> None:
    strict_small = EVAL_SCRIPTS[1].read_text(encoding="utf-8")
    strict = EVAL_SCRIPTS[2].read_text(encoding="utf-8")

    assert 'BACKEND="${3:-${BABYSTEPS_EVAL_BACKEND:-mntp}}"' in strict_small
    assert 'BACKEND="${3:-${BABYSTEPS_EVAL_BACKEND:-causal}}"' in strict
    assert 'FINETUNE_LR="${2:-${BABYSTEPS_SUPERGLUE_LR:-8e-5}}"' in strict_small
    assert 'FINETUNE_LR="${2:-${BABYSTEPS_SUPERGLUE_LR:-3e-5}}"' in strict
    assert 'eval_zero_shot.sh "$MODEL_REPO" "$BACKEND"' in strict_small
    assert 'eval_zero_shot.sh "$MODEL_REPO" "$BACKEND"' in strict
    assert 'collate_preds.sh "${MODEL_REPO##*/}" "$BACKEND"' in strict_small
    assert 'collate_preds.sh "${MODEL_REPO##*/}" "$BACKEND"' in strict


def test_submission_shell_scripts_parse() -> None:
    for script in EVAL_SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize(
    ("track", "expected_revisions"),
    [("strict-small", 19), ("strict", 28)],
)
def test_uploader_dry_run_validates_every_milestone(
    tmp_path: Path,
    track: str,
    expected_revisions: int,
) -> None:
    uploader = _load_script("scripts/hf/upload_babylm_submission.py")
    revisions = uploader.revisions_for_track(track)
    assert len(revisions) == expected_revisions
    _write_checkpoint(tmp_path / "final")
    for revision in revisions:
        _write_checkpoint(tmp_path / revision)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "hf" / "upload_babylm_submission.py"),
            "--run-dir",
            str(tmp_path),
            "--repo-id",
            "example/submission",
            "--track",
            track,
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"validated all {expected_revisions + 1} local checkpoints" in result.stdout
    assert "Visibility on create: private" in result.stdout


def test_bos_preparation_preserves_vocab_and_raw_encodings(tmp_path: Path) -> None:
    backend = Tokenizer(
        models.WordLevel(
            vocab={"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "Hello": 4, "world": 5, ".": 6},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    source_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    source = tmp_path / "source"
    output = tmp_path / "output"
    source_tokenizer.save_pretrained(source)
    raw_before = source_tokenizer("Hello world.", add_special_tokens=False)["input_ids"]

    preparer = _load_script("scripts/tokenization/prepare_submission_bos_tokenizer.py")
    preparer.prepare_tokenizer(source, output)
    prepared = PreTrainedTokenizerFast.from_pretrained(output)

    assert prepared.get_vocab() == source_tokenizer.get_vocab()
    assert prepared("Hello world.", add_special_tokens=False)["input_ids"] == raw_before
    encoded = prepared("Hello world.", return_offsets_mapping=True)
    assert encoded["input_ids"] == [prepared.bos_token_id, *raw_before]
    assert encoded["offset_mapping"][0] == (0, 0)


def test_evaluator_setup_pins_commit_and_is_idempotent_by_construction() -> None:
    setup = EVAL_SCRIPTS[0].read_text(encoding="utf-8")
    patch = (ROOT / "patches" / "babylm-eval" / "compat.patch").read_text(
        encoding="utf-8"
    )

    assert 'EXPECTED_COMMIT="3d57ddc8c40ee795c0b5e41b3a20251a9457a593"' in setup
    assert 'apply --reverse --check "$PATCH_FILE"' in setup
    assert 'apply --check "$PATCH_FILE"' in setup
    assert 'diff --name-only HEAD --' in setup
    assert 'diff --quiet HEAD --' in setup
    assert "strict/models/" not in patch
    assert "strict/results/" not in patch
    assert "diff --git a/.env" not in patch


def test_readme_links_and_command_targets_exist() -> None:
    markdown_files = (ROOT / "README.md", ROOT / "scripts" / "README.md", ROOT / "data" / "README.md", ROOT / "tests" / "README.md")
    markdown_link = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for markdown in markdown_files:
        for target in markdown_link.findall(markdown.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            assert (markdown.parent / target).resolve().exists(), f"Broken link in {markdown}: {target}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    command_target = re.compile(
        r"(?:python|bash) ((?:scripts|src)/[A-Za-z0-9_./*-]+)"
        r"|accelerate launch --num_processes \d+ ((?:scripts|src)/[A-Za-z0-9_./*-]+)"
    )
    for direct_target, accelerate_target in command_target.findall(readme):
        target = direct_target or accelerate_target
        if "*" not in target:
            assert (ROOT / target).is_file(), f"Missing README command target: {target}"
