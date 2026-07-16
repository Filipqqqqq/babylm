from __future__ import annotations

import importlib.util
from pathlib import Path

from babysteps.evaluation.entity_tracking_likelihood import ContinuationScore


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/eval_entity_tracking_multidomain_hf.py"
SPEC = importlib.util.spec_from_file_location("multidomain_eval_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pair_summary_reports_completion_and_symmetric_accuracy() -> None:
    rows = [
        {
            "pair_id": f"pair-{split}",
            "split": split,
            "domain": "zoo",
            "renderer": "narrative",
            "counterfactual_kind": "move",
        }
        for split in MODULE.EVAL_SPLITS
    ]
    scores = {}
    for index in range(2 * len(rows)):
        correct = 3.0 if index % 2 == 0 else 2.0
        wrong = 1.0 if index % 2 == 0 else 3.0
        scores[index] = {
            0: ContinuationScore(correct, 1, 2),
            1: ContinuationScore(wrong, 1, 2),
        }

    records = MODULE.pair_records(rows, scores, "sum")
    summary = MODULE.pair_summary(records)

    assert all(row["completion_accuracy"] == 0.5 for row in summary)
    assert all(row["both_correct_rate"] == 0.0 for row in summary)
    assert all(row["symmetric_accuracy"] == 1.0 for row in summary)
