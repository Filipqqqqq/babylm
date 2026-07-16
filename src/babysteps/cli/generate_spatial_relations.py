#!/usr/bin/env python3
"""Generate synthetic spatial-relation rows from symbolic scenes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from babysteps.paths import expand_env_vars, resolve_path
from babysteps.spatial_synth.generate import (
    DEFAULT_FAMILIES,
    DEFAULT_RANK_OBJECTIVE,
    RankObjective,
    generate_examples,
    generate_examples_from_family_counts,
    write_items,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "data" / "spatial_relations_synth_context_rank.yaml"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f) or {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML config with generator/output settings.",
    )
    parser.add_argument("--output", help="Output path. Suffix chooses JSONL or CSV.")
    parser.add_argument("--format", choices=("jsonl", "csv"), help="Override output format.")
    parser.add_argument("--n", type=int, help="Number of synthetic items to generate.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--objective",
        choices=("completion_rank", "context_rank"),
        help="Rank objective/output schema to generate.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=DEFAULT_FAMILIES,
        help="Template families to include, round-robin balanced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, base_dir=Path.cwd())
    config = load_yaml_config(config_path)
    generator_config = config.get("generator", {})
    output_config = config.get("output", {})

    seed = int(args.seed if args.seed is not None else generator_config.get("seed", 0))
    families = tuple(args.families or generator_config.get("families", DEFAULT_FAMILIES))
    objective: RankObjective = (
        args.objective
        or generator_config.get("objective", DEFAULT_RANK_OBJECTIVE)
    )
    family_counts = generator_config.get("family_counts")
    use_family_counts = family_counts is not None and args.n is None and args.families is None
    count = int(
        args.n
        if args.n is not None
        else (
            sum(int(count) for count in family_counts.values())
            if use_family_counts
            else generator_config.get("n", 1000)
        )
    )

    output_value = args.output or output_config.get("path")
    if not output_value:
        raise ValueError("Set output.path in the config or pass --output.")
    output_path = resolve_path(
        output_value,
        base_dir=Path.cwd() if args.output else ROOT,
    )
    output_format = args.format or output_config.get("format")

    if use_family_counts:
        items = generate_examples_from_family_counts(
            family_counts,
            seed=seed,
            objective=objective,
        )
        family_order = tuple(family_counts)
    else:
        items = generate_examples(count, seed=seed, families=families, objective=objective)
        family_order = families
    write_items(items, output_path, output_format=output_format)

    family_count_report = {family: 0 for family in family_order}
    for item in items:
        family_count_report[item.template_family] = (
            family_count_report.get(item.template_family, 0) + 1
        )

    unit = "pairs" if objective == "context_rank" else "items"
    print(f"Wrote {len(items):,} spatial-relation {unit} to {output_path}")
    print(f"Objective: {objective}")
    print("Families: " + ", ".join(f"{k}={v}" for k, v in family_count_report.items()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
