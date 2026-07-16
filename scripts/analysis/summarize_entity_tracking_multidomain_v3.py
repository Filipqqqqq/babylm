#!/usr/bin/env python3
"""Summarize v3 seed pairs and apply the documented stage gate."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


SPLITS = ("iid", "lexical_ood", "template_ood", "domain_ood", "compositional_ood")
RUN_RE = re.compile(r"seed(?P<seed>\d+)_(?P<arm>baseline|lm_only|completion_rank|symmetric_rank)_")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _co_primary_score(path: Path) -> float:
    candidates = list(path.rglob("likelihood_summary.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one likelihood summary under {path}, found {len(candidates)}.")
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
            raise ValueError(f"Expected one summed five-choice row for {split} under {path}.")
        result[split] = float(matches[0]["score"])
    return result


def collect(experiment_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted((experiment_root / "evaluation").glob("seed*")):
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        row: dict[str, Any] = {
            "seed": int(match.group("seed")),
            "arm": match.group("arm"),
            "custom": _co_primary_score(run_dir / "01_custom_v1"),
            "official": _co_primary_score(run_dir / "02_official"),
        }
        row.update(_multidomain_scores(run_dir / "03_multidomain_v3"))
        rows.append(row)
    return rows


def initial_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["seed"], row["arm"]): row for row in rows}
    seeds = sorted(
        seed
        for seed in {int(row["seed"]) for row in rows}
        if (seed, "baseline") in by_key and (seed, "lm_only") in by_key
    )
    if len(seeds) != 3:
        return {"ready": False, "passed": False, "reason": "three paired seeds are not complete"}
    deltas = {
        metric: [
            float(by_key[(seed, "lm_only")][metric])
            - float(by_key[(seed, "baseline")][metric])
            for seed in seeds
        ]
        for metric in ("custom", "official", *SPLITS)
    }
    means = {metric: statistics.mean(values) for metric, values in deltas.items()}
    conditions = {
        "mean_iid_positive": means["iid"] > 0.0,
        "two_of_three_iid_nonnegative": sum(value >= 0.0 for value in deltas["iid"]) >= 2,
        "custom_no_worse_than_minus_1pp": means["custom"] >= -0.01,
        "official_no_worse_than_minus_1pp": means["official"] >= -0.01,
    }
    return {
        "ready": True,
        "passed": all(conditions.values()),
        "seeds": seeds,
        "conditions": conditions,
        "mean_deltas": means,
        "seed_deltas": {metric: dict(zip(map(str, seeds), values)) for metric, values in deltas.items()},
        "next_stage": "completion_rank" if all(conditions.values()) else "stop",
    }


def write_outputs(experiment_root: Path, rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    fields = ("seed", "arm", "custom", "official", *SPLITS)
    with (experiment_root / "initial_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    with (experiment_root / "initial_gate.json").open("w", encoding="utf-8") as f:
        json.dump(gate, f, indent=2, sort_keys=True)
        f.write("\n")
    with (experiment_root / "initial_results.md").open("w", encoding="utf-8") as f:
        f.write("# Multi-domain v3 initial results\n\n")
        f.write("| Seed | Arm | Custom | Official | IID | Lexical OOD | Template OOD | Domain OOD | Compositional OOD |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            values = [f"{float(row[key]):.4f}" for key in ("custom", "official", *SPLITS)]
            f.write(f"| {row['seed']} | {row['arm']} | " + " | ".join(values) + " |\n")
        f.write("\n## Gate\n\n")
        f.write(f"Ready: `{gate['ready']}`  \nPassed: `{gate['passed']}`  \n")
        f.write(f"Next stage: `{gate.get('next_stage', 'pending')}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    rows = collect(args.experiment_root)
    gate = initial_gate(rows)
    write_outputs(args.experiment_root, rows, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
