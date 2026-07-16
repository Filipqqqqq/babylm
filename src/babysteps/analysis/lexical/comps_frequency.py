"""COMPS lexical target-frequency plots."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


FrequencySlice = Literal["top", "middle", "bottom"]
FREQUENCY_SLICES: tuple[FrequencySlice, ...] = ("top", "middle", "bottom")
EXPOSURE_BUCKET_ORDER = (
    "0",
    "1-5",
    "6-10",
    "11-50",
    "51-100",
    "101-500",
    "501-1000",
    "1001+",
)


@dataclass(frozen=True)
class FrequencyTarget:
    """One target-level frequency row from lexical target coverage."""

    target_id: str
    label: str
    frequency: float
    document_count: int
    exposure_bucket: str
    rank: int


def load_frequency_targets(
    target_coverage_csv: str | Path,
    *,
    frequency_column: str = "total_mentions",
) -> list[FrequencyTarget]:
    """Load and rank COMPS target frequencies from target_coverage.csv."""

    rows = []
    with Path(target_coverage_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "target_id": str(row.get("target_id", "")),
                    "label": str(row.get("label", row.get("target_id", ""))),
                    "frequency": _float_value(row.get(frequency_column, "0")),
                    "document_count": _int_value(row.get("document_count", "0")),
                    "exposure_bucket": str(row.get("exposure_bucket", "")),
                }
            )

    rows.sort(key=lambda row: (-float(row["frequency"]), str(row["label"])))
    return [
        FrequencyTarget(
            target_id=str(row["target_id"]),
            label=str(row["label"]),
            frequency=float(row["frequency"]),
            document_count=int(row["document_count"]),
            exposure_bucket=str(row["exposure_bucket"]),
            rank=index + 1,
        )
        for index, row in enumerate(rows)
    ]


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def select_frequency_slice(
    targets: list[FrequencyTarget],
    *,
    slice_name: FrequencySlice,
    n: int = 30,
) -> list[FrequencyTarget]:
    """Select top, middle, or bottom targets from ranked frequency rows."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if slice_name not in FREQUENCY_SLICES:
        raise ValueError(f"Unsupported frequency slice: {slice_name!r}.")
    if not targets:
        return []

    count = min(n, len(targets))
    if slice_name == "top":
        return targets[:count]
    if slice_name == "middle":
        start = max(0, (len(targets) - count) // 2)
        return targets[start : start + count]

    bottom = sorted(targets, key=lambda target: (target.frequency, target.label))[:count]
    return sorted(bottom, key=lambda target: (-target.frequency, target.label))


def build_frequency_slice_rows(
    targets: list[FrequencyTarget],
    *,
    n: int = 30,
) -> list[dict[str, Any]]:
    """Build CSV/report rows for top, middle, and bottom target slices."""

    rows: list[dict[str, Any]] = []
    for slice_name in FREQUENCY_SLICES:
        for slice_rank, target in enumerate(
            select_frequency_slice(targets, slice_name=slice_name, n=n),
            start=1,
        ):
            rows.append(
                {
                    "slice": slice_name,
                    "slice_rank": slice_rank,
                    "overall_rank": target.rank,
                    "target_id": target.target_id,
                    "label": target.label,
                    "frequency": target.frequency,
                    "document_count": target.document_count,
                    "exposure_bucket": target.exposure_bucket,
                }
            )
    return rows


def write_frequency_slice_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slice",
        "slice_rank",
        "overall_rank",
        "target_id",
        "label",
        "frequency",
        "document_count",
        "exposure_bucket",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_frequency_slices(
    targets: list[FrequencyTarget],
    output_dir: str | Path,
    *,
    n: int = 30,
    image_format: str = "png",
    log_x: bool = False,
    title_prefix: str = "COMPS lexical",
    x_label: str = "Total mentions",
) -> list[Path]:
    """Write combined and per-slice top/middle/bottom frequency plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    slice_rows = {
        slice_name: select_frequency_slice(targets, slice_name=slice_name, n=n)
        for slice_name in FREQUENCY_SLICES
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6.0, 0.26 * n + 2.5)))
    for ax, slice_name in zip(axes, FREQUENCY_SLICES):
        _plot_slice_axis(
            ax,
            slice_rows[slice_name],
            title=_slice_title(slice_name, n=n),
            log_x=log_x,
            x_label=x_label,
        )
    fig.suptitle(f"{title_prefix} target frequencies", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    combined_path = output_dir / f"frequency_top_middle_bottom.{image_format}"
    fig.savefig(combined_path, dpi=160)
    plt.close(fig)
    paths.append(combined_path)

    for slice_name in FREQUENCY_SLICES:
        fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.28 * n + 2.0)))
        _plot_slice_axis(
            ax,
            slice_rows[slice_name],
            title=f"{title_prefix} {_slice_title(slice_name, n=n).lower()}",
            log_x=log_x,
            x_label=x_label,
        )
        fig.tight_layout()
        path = output_dir / f"frequency_{slice_name}.{image_format}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def _slice_title(slice_name: str, *, n: int) -> str:
    if slice_name == "top":
        return f"Top {n}"
    if slice_name == "middle":
        return f"Middle {n}"
    return f"Bottom {n}"


def _plot_slice_axis(
    ax,
    rows: list[FrequencyTarget],
    *,
    title: str,
    log_x: bool,
    x_label: str,
) -> None:
    labels = [_short_label(row.label) for row in reversed(rows)]
    values = [row.frequency for row in reversed(rows)]
    y_positions = range(len(rows))
    ax.barh(list(y_positions), values, color="#4c78a8")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    if log_x and any(value > 0 for value in values):
        positive_min = min(value for value in values if value > 0)
        ax.set_xscale("log")
        ax.set_xlim(left=max(0.5, positive_min * 0.8))


def _short_label(label: str, *, max_chars: int = 42) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 3] + "..."


def plot_exposure_buckets(
    targets: list[FrequencyTarget],
    output_dir: str | Path,
    *,
    image_format: str = "png",
    title_prefix: str = "COMPS lexical",
) -> Path:
    """Write an exposure-bucket count plot for the full target inventory."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(target.exposure_bucket for target in targets)
    labels = [bucket for bucket in EXPOSURE_BUCKET_ORDER if bucket in counts]
    labels.extend(sorted(bucket for bucket in counts if bucket not in EXPOSURE_BUCKET_ORDER))
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(labels, values, color="#59a14f")
    ax.set_title(f"{title_prefix} exposure buckets")
    ax.set_xlabel("Exposure bucket")
    ax.set_ylabel("Target count")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    path = output_dir / f"exposure_buckets.{image_format}"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_comps_frequency_report(
    *,
    target_coverage_csv: str | Path,
    output_dir: str | Path,
    frequency_column: str = "total_mentions",
    n: int = 30,
    image_format: str = "png",
    log_x: bool = False,
    title_prefix: str = "COMPS lexical",
) -> dict[str, Any]:
    """Build COMPS lexical frequency CSVs and plots."""

    targets = load_frequency_targets(
        target_coverage_csv,
        frequency_column=frequency_column,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_rows = build_frequency_slice_rows(targets, n=n)
    csv_path = output_dir / "frequency_top_middle_bottom.csv"
    write_frequency_slice_csv(slice_rows, csv_path)

    plot_paths = plot_frequency_slices(
        targets,
        output_dir,
        n=n,
        image_format=image_format,
        log_x=log_x,
        title_prefix=title_prefix,
        x_label=frequency_column.replace("_", " ").title(),
    )
    plot_paths.append(
        plot_exposure_buckets(
            targets,
            output_dir,
            image_format=image_format,
            title_prefix=title_prefix,
        )
    )

    return {
        "csv": csv_path,
        "plots": plot_paths,
        "target_count": len(targets),
        "n": int(n),
        "frequency_column": frequency_column,
    }


__all__ = [
    "EXPOSURE_BUCKET_ORDER",
    "FREQUENCY_SLICES",
    "FrequencyTarget",
    "build_comps_frequency_report",
    "build_frequency_slice_rows",
    "load_frequency_targets",
    "plot_exposure_buckets",
    "plot_frequency_slices",
    "select_frequency_slice",
    "write_frequency_slice_csv",
]
