"""Report writers and plots for Word2Vec EWoK analyses."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from babysteps.analysis.word2vec.ewok_eval import EwokScoreRecord


ITEM_SCORE_FIELDS = [
    "item_id",
    "domain",
    "concept_a",
    "concept_b",
    "context_type",
    "context_diff",
    "target_diff",
    "source_file",
    "line_number",
    "status",
    "context_sensitivity",
    "k1",
    "k2",
    "acc1",
    "acc2",
    "S11",
    "S21",
    "S22",
    "S12",
    "c1_tokens_total",
    "c1_tokens_in_vocab",
    "c1_retained_fraction",
    "t1_tokens_total",
    "t1_tokens_in_vocab",
    "t1_retained_fraction",
    "c2_tokens_total",
    "c2_tokens_in_vocab",
    "c2_retained_fraction",
    "t2_tokens_total",
    "t2_tokens_in_vocab",
    "t2_retained_fraction",
    "context_1",
    "target_1",
    "context_2",
    "target_2",
]

DOMAIN_SUMMARY_FIELDS = [
    "domain",
    "items_total",
    "items_counted",
    "items_insufficient_vectors",
    "context_sensitivity_mean",
    "k1_mean",
    "k2_mean",
    "abs_margin_mean",
    "span_retained_fraction_mean",
]

OOV_SUMMARY_FIELDS = [
    "token",
    "count",
    "domains",
    "spans",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return value


def _format_float(value: float, *, digits: int = 6) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _write_csv(rows: list[dict[str, Any]], path: str | Path, *, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _pooled_fields(prefix: str, record: EwokScoreRecord) -> dict[str, str | int]:
    pooled = getattr(record, prefix)
    return {
        f"{prefix}_tokens_total": pooled.total_tokens,
        f"{prefix}_tokens_in_vocab": pooled.in_vocab_count,
        f"{prefix}_retained_fraction": _format_float(pooled.retained_fraction),
    }


def item_score_rows(records: list[EwokScoreRecord]) -> list[dict[str, Any]]:
    """Build per-item rows with raw scores and OOV diagnostics."""

    rows = []
    for record in records:
        item = record.item
        row: dict[str, Any] = {
            "item_id": item.item_id,
            "domain": item.domain,
            "concept_a": item.concept_a,
            "concept_b": item.concept_b,
            "context_type": item.context_type,
            "context_diff": item.context_diff,
            "target_diff": item.target_diff,
            "source_file": item.source_file,
            "line_number": item.line_number,
            "status": record.status,
            "context_sensitivity": _format_float(record.context_sensitivity),
            "k1": _format_float(record.k1),
            "k2": _format_float(record.k2),
            "acc1": _format_float(record.acc1),
            "acc2": _format_float(record.acc2),
            "S11": _format_float(record.s11),
            "S21": _format_float(record.s21),
            "S22": _format_float(record.s22),
            "S12": _format_float(record.s12),
            "context_1": item.context_1,
            "target_1": item.target_1,
            "context_2": item.context_2,
            "target_2": item.target_2,
        }
        for prefix in ("c1", "t1", "c2", "t2"):
            row.update(_pooled_fields(prefix, record))
        rows.append(row)
    return rows


def domain_summary_rows(records: list[EwokScoreRecord]) -> list[dict[str, Any]]:
    """Summarize context sensitivity and vector coverage by EWoK domain."""

    grouped: dict[str, list[EwokScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[record.item.domain].append(record)

    rows = []
    for domain in sorted(grouped):
        domain_records = grouped[domain]
        counted = [record for record in domain_records if record.is_counted]
        context_values = [
            record.context_sensitivity
            for record in counted
            if not math.isnan(record.context_sensitivity)
        ]
        k1_values = [record.k1 for record in counted if not math.isnan(record.k1)]
        k2_values = [record.k2 for record in counted if not math.isnan(record.k2)]
        abs_margins = [abs(value) for value in k1_values + k2_values]
        retained = []
        for record in domain_records:
            retained.extend(
                [
                    record.c1.retained_fraction,
                    record.t1.retained_fraction,
                    record.c2.retained_fraction,
                    record.t2.retained_fraction,
                ]
            )
        rows.append(
            {
                "domain": domain,
                "items_total": len(domain_records),
                "items_counted": len(counted),
                "items_insufficient_vectors": len(domain_records) - len(counted),
                "context_sensitivity_mean": _format_float(_mean(context_values)),
                "k1_mean": _format_float(_mean(k1_values)),
                "k2_mean": _format_float(_mean(k2_values)),
                "abs_margin_mean": _format_float(_mean(abs_margins)),
                "span_retained_fraction_mean": _format_float(_mean(retained)),
            }
        )
    return rows


def oov_summary_rows(records: list[EwokScoreRecord], *, limit: int = 200) -> list[dict[str, Any]]:
    """Build frequent OOV-token diagnostics across items and spans."""

    counts: Counter[str] = Counter()
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    spans: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for span_name, pooled in {
            "c1": record.c1,
            "t1": record.t1,
            "c2": record.c2,
            "t2": record.t2,
        }.items():
            for token in pooled.oov_tokens:
                counts[token] += 1
                domains[token][record.item.domain] += 1
                spans[token][span_name] += 1

    rows = []
    for token, count in counts.most_common(limit):
        rows.append(
            {
                "token": token,
                "count": count,
                "domains": "; ".join(
                    f"{domain}:{domain_count}"
                    for domain, domain_count in domains[token].most_common(5)
                ),
                "spans": "; ".join(
                    f"{span}:{span_count}" for span, span_count in spans[token].most_common()
                ),
            }
        )
    return rows


def plot_domain_context_sensitivity(
    domain_rows: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Plot mean paper-style context sensitivity by domain."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        domain_rows,
        key=lambda row: float(row["context_sensitivity_mean"])
        if row["context_sensitivity_mean"] != "NaN"
        else -1.0,
        reverse=True,
    )
    domains = [str(row["domain"]) for row in rows]
    values = [
        0.0 if row["context_sensitivity_mean"] == "NaN" else float(row["context_sensitivity_mean"])
        for row in rows
    ]
    fig_width = max(8.0, min(18.0, 0.55 * len(domains) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    bars = ax.bar(range(len(domains)), values, color="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax.set_title("EWoK Word2Vec context sensitivity by domain")
    ax.set_ylabel("Context sensitivity")
    ax.set_xlabel("Domain")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, row in zip(bars, rows):
        label = f"n={row['items_counted']}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#888888",
            fontsize=8,
        )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_margin_distribution(records: list[EwokScoreRecord], path: str | Path) -> Path:
    """Plot k1/k2 margin distributions for counted EWoK items."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k1_values = [record.k1 for record in records if record.is_counted and not math.isnan(record.k1)]
    k2_values = [record.k2 for record in records if record.is_counted and not math.isnan(record.k2)]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if k1_values:
        ax.hist(k1_values, bins=40, alpha=0.65, label="k1 = S11 - S21", color="#4c78a8")
    if k2_values:
        ax.hist(k2_values, bins=40, alpha=0.65, label="k2 = S22 - S12", color="#f58518")
    ax.axvline(0, color="#444444", linewidth=1.0)
    ax.set_title("EWoK Word2Vec context-sensitivity margins")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Item count")
    ax.legend()
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_eval_report(
    records: list[EwokScoreRecord],
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    make_plots: bool = True,
) -> dict[str, Path]:
    """Write item, domain, OOV, manifest, and plot artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    item_rows = item_score_rows(records)
    domain_rows = domain_summary_rows(records)
    oov_rows = oov_summary_rows(records)

    item_path = output_dir / "item_scores.csv"
    domain_path = output_dir / "domain_summary.csv"
    oov_path = output_dir / "oov_summary.csv"
    manifest_path = output_dir / "manifest.json"
    _write_csv(item_rows, item_path, fieldnames=ITEM_SCORE_FIELDS)
    _write_csv(domain_rows, domain_path, fieldnames=DOMAIN_SUMMARY_FIELDS)
    _write_csv(oov_rows, oov_path, fieldnames=OOV_SUMMARY_FIELDS)

    counted_context = [
        record.context_sensitivity
        for record in records
        if record.is_counted and not math.isnan(record.context_sensitivity)
    ]
    manifest = dict(manifest)
    manifest["items_total"] = len(records)
    manifest["items_counted"] = len(counted_context)
    manifest["context_sensitivity_mean"] = _mean(counted_context)
    if counted_context:
        manifest["context_sensitivity_median"] = statistics.median(counted_context)
    _write_json(manifest, manifest_path)

    paths = {
        "item_scores": item_path,
        "domain_summary": domain_path,
        "oov_summary": oov_path,
        "manifest": manifest_path,
    }
    if make_plots:
        paths["domain_context_sensitivity_plot"] = plot_domain_context_sensitivity(
            domain_rows,
            output_dir / "domain_context_sensitivity.png",
        )
        paths["margin_distribution_plot"] = plot_margin_distribution(
            records,
            output_dir / "margin_distribution.png",
        )
    return paths


def read_domain_summary(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compare_domain_summaries(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    output_dir: str | Path,
    make_plots: bool = True,
) -> dict[str, Path]:
    """Compare two Word2Vec EWoK domain summaries."""

    left_by_domain = {str(row["domain"]): row for row in left_rows}
    right_by_domain = {str(row["domain"]): row for row in right_rows}
    rows = []
    for domain in sorted(set(left_by_domain) | set(right_by_domain)):
        left_value = _summary_value(left_by_domain.get(domain), "context_sensitivity_mean")
        right_value = _summary_value(right_by_domain.get(domain), "context_sensitivity_mean")
        delta = (
            math.nan
            if math.isnan(left_value) or math.isnan(right_value)
            else right_value - left_value
        )
        rows.append(
            {
                "domain": domain,
                f"{left_label}_context_sensitivity": _format_float(left_value),
                f"{right_label}_context_sensitivity": _format_float(right_value),
                f"{right_label}_minus_{left_label}": _format_float(delta),
                f"{left_label}_items_counted": _summary_text(left_by_domain.get(domain), "items_counted"),
                f"{right_label}_items_counted": _summary_text(right_by_domain.get(domain), "items_counted"),
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "domain_context_sensitivity_comparison.csv"
    fieldnames = list(rows[0].keys()) if rows else ["domain"]
    _write_csv(rows, csv_path, fieldnames=fieldnames)
    paths = {"comparison_csv": csv_path}
    if make_plots:
        paths["comparison_plot"] = plot_domain_comparison(
            rows,
            left_label=left_label,
            right_label=right_label,
            output_path=output_dir / "domain_context_sensitivity_comparison.png",
        )
    return paths


def _summary_value(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return math.nan
    try:
        return float(row.get(key, "NaN"))
    except (TypeError, ValueError):
        return math.nan


def _summary_text(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return ""
    return str(row.get(key, ""))


def plot_domain_comparison(
    rows: list[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    output_path: str | Path,
) -> Path:
    """Plot two domain-summary runs side by side."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = [str(row["domain"]) for row in rows]
    left_key = f"{left_label}_context_sensitivity"
    right_key = f"{right_label}_context_sensitivity"
    left_values = [_safe_plot_value(row[left_key]) for row in rows]
    right_values = [_safe_plot_value(row[right_key]) for row in rows]

    fig_width = max(8.0, min(20.0, 0.65 * len(domains) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    xs = list(range(len(domains)))
    width = 0.38
    ax.bar([x - width / 2 for x in xs], left_values, width=width, label=left_label, color="#4c78a8")
    ax.bar([x + width / 2 for x in xs], right_values, width=width, label=right_label, color="#f58518")
    ax.set_ylim(0, 1.05)
    ax.set_title("EWoK Word2Vec context sensitivity comparison")
    ax.set_ylabel("Context sensitivity")
    ax.set_xlabel("Domain")
    ax.set_xticks(xs)
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _safe_plot_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


__all__ = [
    "DOMAIN_SUMMARY_FIELDS",
    "ITEM_SCORE_FIELDS",
    "OOV_SUMMARY_FIELDS",
    "compare_domain_summaries",
    "domain_summary_rows",
    "item_score_rows",
    "oov_summary_rows",
    "read_domain_summary",
    "write_eval_report",
]
