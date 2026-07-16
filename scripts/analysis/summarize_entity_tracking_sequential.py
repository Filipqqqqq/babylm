#!/usr/bin/env python3
"""Summarize paired regular-mixing and sequential entity-tracking results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SPLITS = ("iid", "lexical_ood", "template_ood", "domain_ood", "compositional_ood")
METRICS = ("custom", "official", *SPLITS)
STAGES = ("baseline", "regular_mix", "midtrain", "finetune")
COMPARISONS = (
    ("regular_mix_minus_baseline", "regular_mix", "baseline"),
    ("midtrain_minus_baseline", "midtrain", "baseline"),
    ("midtrain_minus_regular_mix", "midtrain", "regular_mix"),
    ("finetune_minus_midtrain", "finetune", "midtrain"),
    ("finetune_minus_regular_mix", "finetune", "regular_mix"),
    ("finetune_minus_baseline", "finetune", "baseline"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _co_primary_score(path: Path) -> float:
    candidates = list(path.rglob("likelihood_summary.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one likelihood summary under {path}, found {len(candidates)}."
        )
    for row in _read_csv(candidates[0]):
        if (
            row.get("benchmark") == "entity_tracking"
            and row.get("group_scope") == "overall"
            and row.get("group_name") == "all"
            and row.get("score_reduction") == "sum"
        ):
            return float(row["score"])
    raise ValueError(f"No summed Entity Tracking overall row in {candidates[0]}.")


def _multidomain_scores(path: Path) -> dict[str, float]:
    rows = _read_csv(path / "five_choice_summary.csv")
    result = {}
    for split in SPLITS:
        matches = [
            row
            for row in rows
            if row.get("group_scope") == "family"
            and row.get("group_name") == split
            and row.get("score_reduction") == "sum"
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one summed row for {split} under {path}.")
        result[split] = float(matches[0]["score"])
    return result


def _scores(eval_dir: Path) -> dict[str, float]:
    values = {
        "custom": _co_primary_score(eval_dir / "01_custom_v1"),
        "official": _co_primary_score(eval_dir / "02_official"),
    }
    values.update(_multidomain_scores(eval_dir / "03_multidomain_v3"))
    return values


def collect(reference_root: Path, experiment_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        reference_eval = reference_root / "evaluation"
        stage_dirs = {
            "baseline": reference_eval / f"seed{seed}_baseline_matched_local_baseline",
            "regular_mix": reference_eval / f"seed{seed}_lm_only_multidomain_v3_lm_only",
            "midtrain": experiment_root / "evaluation" / f"seed{seed}_midtrain",
            "finetune": experiment_root / "evaluation" / f"seed{seed}_finetune",
        }
        for stage, eval_dir in stage_dirs.items():
            if not eval_dir.exists():
                continue
            rows.append({"seed": seed, "stage": stage, **_scores(eval_dir)})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(int(row["seed"]), str(row["stage"])): row for row in rows}
    complete_seeds = [
        seed
        for seed in (42, 43, 44)
        if all((seed, stage) in by_key for stage in STAGES)
    ]
    stage_means = {
        stage: {
            metric: statistics.mean(
                float(by_key[(seed, stage)][metric]) for seed in complete_seeds
            )
            for metric in METRICS
        }
        for stage in STAGES
    } if complete_seeds else {}
    deltas: dict[str, Any] = {}
    for name, treatment, control in COMPARISONS:
        seed_deltas = {
            str(seed): {
                metric: float(by_key[(seed, treatment)][metric])
                - float(by_key[(seed, control)][metric])
                for metric in METRICS
            }
            for seed in complete_seeds
        }
        deltas[name] = {
            "mean": {
                metric: statistics.mean(
                    seed_deltas[str(seed)][metric] for seed in complete_seeds
                )
                for metric in METRICS
            } if complete_seeds else {},
            "by_seed": seed_deltas,
        }
    return {
        "ready": len(complete_seeds) == 3,
        "complete_seeds": complete_seeds,
        "stage_means": stage_means,
        "comparisons": deltas,
    }


def write_outputs(experiment_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fields = ("seed", "stage", *METRICS)
    with (experiment_root / "sequential_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    with (experiment_root / "sequential_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (experiment_root / "sequential_results.md").open("w", encoding="utf-8") as f:
        f.write("# Entity-tracking sequential timing results\n\n")
        f.write("| Seed | Stage | Custom | Official | IID | Lexical OOD | Template OOD | Domain OOD | Compositional OOD |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            values = " | ".join(f"{float(row[metric]):.4f}" for metric in METRICS)
            f.write(f"| {row['seed']} | {row['stage']} | {values} |\n")
        if summary["ready"]:
            f.write("\n## Paired mean deltas\n\n")
            f.write("| Comparison | " + " | ".join(METRICS) + " |\n")
            f.write("|---|" + "---:|" * len(METRICS) + "\n")
            for name, _, _ in COMPARISONS:
                means = summary["comparisons"][name]["mean"]
                values = " | ".join(f"{float(means[metric]):+.4f}" for metric in METRICS)
                f.write(f"| {name} | {values} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "/home/jorge/babylm-baby-steps-runs/experiments/"
            "entity_tracking_multidomain_v3/20260716_092451"
        ),
    )
    args = parser.parse_args()
    rows = collect(args.reference_root, args.experiment_root)
    summary = summarize(rows)
    write_outputs(args.experiment_root, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
