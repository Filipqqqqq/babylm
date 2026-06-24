from __future__ import annotations

import json
from pathlib import Path

from babysteps.analysis.loss_dynamics import load_training_loss_rows
from babysteps.analysis.loss_dynamics import write_training_loss_plots


def write_metrics(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def test_load_training_loss_rows_reads_causal_and_hybrid_metrics(tmp_path: Path) -> None:
    causal_metrics = write_metrics(
        tmp_path / "llama_muon" / "metrics_history.json",
        [
            {"type": "train", "step": 1, "words_seen": 100.0, "loss": 4.0},
            {"type": "validation", "step": 1, "words_seen": 100.0, "loss": 4.2},
        ],
    )
    hybrid_metrics = write_metrics(
        tmp_path / "gpt_bert_lamb" / "metrics_history.json",
        [
            {
                "type": "train",
                "step": 1,
                "words_seen": 100.0,
                "loss": 5.0,
                "train_causal_loss": 4.8,
                "train_mntp_loss": 5.2,
            }
        ],
    )

    rows = load_training_loss_rows(
        [causal_metrics, hybrid_metrics],
        labels=["llama+muon", "gpt-bert+lamb"],
        loss_keys=("loss", "train_causal_loss", "train_mntp_loss"),
    )

    assert {(row["series"], row["metric"]) for row in rows} == {
        ("llama+muon", "loss"),
        ("gpt-bert+lamb", "loss"),
        ("gpt-bert+lamb", "train_causal_loss"),
        ("gpt-bert+lamb", "train_mntp_loss"),
    }
    assert all(row["x"] == 100.0 for row in rows)


def test_write_training_loss_plots_writes_one_png_per_available_metric(tmp_path: Path) -> None:
    metrics = write_metrics(
        tmp_path / "run" / "metrics_history.json",
        [
            {"type": "train", "step": 1, "words_seen": 100.0, "loss": 4.0},
            {"type": "train", "step": 2, "words_seen": 200.0, "loss": 3.8},
        ],
    )

    written = write_training_loss_plots(
        metrics_paths=[metrics],
        output_dir=tmp_path / "plots",
        loss_keys=("loss", "train_causal_loss"),
    )

    assert written == [tmp_path / "plots" / "loss_by_words_seen.png"]
    assert written[0].exists()
