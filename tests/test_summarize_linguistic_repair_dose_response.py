from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analysis" / "summarize_linguistic_repair_dose_response.py"
SPEC = importlib.util.spec_from_file_location("summarize_repair_dose", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_summary(
    root: Path,
    *,
    condition: str,
    pretrain_seed: int,
    learning_rate: str,
    macro: float,
    rte: float,
) -> None:
    path = (
        root
        / "finetuning"
        / condition
        / f"pretrain_seed_{pretrain_seed}"
        / f"lr_{learning_rate}"
        / "finetune_seed_42"
        / "model"
        / "results"
        / "superglue_summary.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "macro_primary": macro,
                "task_results": [
                    {
                        "task": "rte",
                        "primary_metric": "accuracy",
                        "primary_score": rte,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_collect_and_pair_dose_response_results(tmp_path) -> None:
    write_summary(
        tmp_path,
        condition="repair_0pct",
        pretrain_seed=42,
        learning_rate="3e-5",
        macro=0.70,
        rte=0.65,
    )
    write_summary(
        tmp_path,
        condition="repair_1pct",
        pretrain_seed=42,
        learning_rate="3e-5",
        macro=0.68,
        rte=0.60,
    )

    macro_rows, task_rows = MODULE.collect_results(tmp_path)
    paired = MODULE.paired_differences(macro_rows, task_rows)

    assert len(macro_rows) == 2
    assert len(task_rows) == 2
    assert {(row["task"], round(row["difference"], 4)) for row in paired} == {
        ("macro", -0.02),
        ("rte", -0.05),
    }
