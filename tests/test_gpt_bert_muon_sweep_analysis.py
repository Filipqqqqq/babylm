from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "plot_gpt_bert_muon_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_gpt_bert_muon_sweep", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_run(
    root: Path,
    *,
    run_name: str,
    seed: int,
    muon_lr: float,
) -> Path:
    run_dir = root / run_name / "20260629_120000"
    run_dir.mkdir(parents=True)
    config = {
        "run": {"name": run_name, "seed": seed},
        "model": {
            "architecture": "gpt_bert",
            "repo": "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed",
        },
        "training": {"optimizer": "muon_pe", "muon_lr": muon_lr},
    }
    history = [
        {
            "type": "train",
            "epoch": 1,
            "step": 10,
            "tokens_seen": 100,
            "loss": 4.0,
            "train_causal_loss": 3.8,
            "train_mntp_loss": 4.2,
        },
        {
            "type": "validation",
            "epoch": 1,
            "step": 250,
            "tokens_seen": 2500,
            "val_causal_loss": 3.7,
            "val_mntp_loss": 4.1,
        },
        {
            "type": "mntp_likelihood",
            "benchmark": "entity_tracking",
            "score_reduction": "sum",
            "epoch": 5,
            "step": 500,
            "score": 0.45,
        },
        {
            "type": "mntp_likelihood",
            "benchmark": "entity_tracking",
            "score_reduction": "sum",
            "epoch": 10,
            "step": 1000,
            "score": 0.55,
        },
        {
            "type": "ewok_context_sensitivity_full",
            "score_reduction": "sum",
            "epoch": 10,
            "step": 1000,
            "score": 0.65,
        },
    ]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_collect_metric_rows_tags_midpoint_and_end_entity_tracking(tmp_path: Path) -> None:
    module = load_module()
    run_name = "gpt_bert_baseline_mixed_babylm2026_strict_small_muon_lr0p01_seed42"
    write_run(tmp_path, run_name=run_name, seed=42, muon_lr=0.01)

    runs = module.discover_runs(
        tmp_path,
        run_prefix="gpt_bert_baseline_mixed_babylm2026_strict_small_muon",
    )
    rows = module.collect_metric_rows(runs)
    entity_rows = [
        row
        for row in rows
        if module.metric_id(row) == "mntp_likelihood/entity_tracking/sum/score"
    ]

    assert len(runs) == 1
    assert runs[0].muon_lr == 0.01
    assert runs[0].seed == 42
    assert [(row["step"], row["phase"], row["value"]) for row in entity_rows] == [
        (500, "midpoint", 0.45),
        (1000, "end", 0.55),
    ]


def test_run_and_lr_summaries_include_latest_and_phase_columns(tmp_path: Path) -> None:
    module = load_module()
    run_name = "gpt_bert_baseline_mixed_babylm2026_strict_small_muon_lr0p005_seed43"
    write_run(tmp_path, run_name=run_name, seed=43, muon_lr=0.005)
    rows = module.collect_metric_rows(
        module.discover_runs(
            tmp_path,
            run_prefix="gpt_bert_baseline_mixed_babylm2026_strict_small_muon",
        )
    )

    run_summary = module.build_run_summary(rows)
    lr_summary = module.build_lr_summary(rows)

    assert run_summary[0]["latest_validation_val_mntp_loss"] == 4.1
    assert run_summary[0]["midpoint_mntp_likelihood_entity_tracking_sum_score"] == 0.45
    assert run_summary[0]["end_mntp_likelihood_entity_tracking_sum_score"] == 0.55
    assert any(
        row["metric"] == "mntp_likelihood/entity_tracking/sum/score"
        and row["phase"] == "end"
        and row["mean"] == 0.55
        for row in lr_summary
    )
